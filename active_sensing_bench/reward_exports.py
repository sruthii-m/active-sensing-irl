"""Frozen, reloadable reward definitions; IQ includes its training sensor kernel."""
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .belief_model import decode_state
from .data import SCHEMA
from .env import EmptyRoomModel
from .networks import mlp, soft_value
from .state_views import validate_decision_state


def code_version():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
            ("state_views.py", "belief_model.py", "env.py", "networks.py", "reward_exports.py")}


class FrozenReward:
    def __init__(self, network, *, method, representation, view_size=3, gamma=.99, temperature=.001):
        if method not in ("iq", "airl") or representation not in ("full", "belief"):
            raise ValueError("Unsupported reward domain or method")
        self.network = copy.deepcopy(network).cpu().eval().requires_grad_(False)
        self.metadata = dict(schema=SCHEMA, method=method, domain=representation,
                             view_size=view_size, gamma=gamma, temperature=temperature,
                             preprocessing="identity; pose/time normalized by schema",
                             terminal="goal entry or intrinsic deadline 400",
                             kernel="frozen training sensor" if method == "iq" else None,
                             code_version=code_version())
        self.model = EmptyRoomModel(view_size) if method == "iq" else None
        self._successors = lru_cache(maxsize=16384)(self._compute_successors)

    def _compute_successors(self, state_bytes, action):
        return self.model.successors(np.frombuffer(state_bytes, dtype=np.float32), action)

    @torch.no_grad()
    def __call__(self, x, action, next_x=None, task_terminal=None):
        x = np.asarray(x, dtype=np.float32)
        scalar = x.ndim == 1
        x = np.atleast_2d(x)
        action = np.asarray(action)
        if action.ndim == 0:
            action = np.repeat(action, len(x))
        if action.shape != (len(x),) or not np.equal(action, action.astype(int)).all() or ((action < 0) | (action > 6)).any():
            raise ValueError("Invalid reward actions")
        action = action.astype(int)
        for state in x:
            validate_decision_state(state)
        if self.metadata["domain"] == "full" and not np.isin(x[:, 7:], [0, 1]).all():
            raise ValueError("Full reward requires full-state inputs")
        absorbing = np.asarray([state[6] == 1 or state[7 + (decode_state(state)[0][0] - 1) * 8 + decode_state(state)[0][1] - 1] == 1 for state in x])
        inputs = torch.from_numpy(x)
        actions = torch.as_tensor(action, dtype=torch.long)
        if self.metadata["method"] == "airl":
            one_hot = torch.nn.functional.one_hot(actions, 7).float()
            result = self.network(torch.cat((inputs, one_hot), dim=1)).squeeze(1).numpy()
        else:
            result = self.network(inputs).gather(1, actions[:, None]).squeeze(1).numpy().copy()
            states, weights, owners = [], [], []
            for i, state in enumerate(x):
                if absorbing[i]:
                    continue
                for probability, successor, terminal in self._successors(state.tobytes(), int(action[i])):
                    if not terminal:
                        owners.append(i)
                        weights.append(probability)
                        states.append(successor)
            if states:
                values = soft_value(self.network(torch.from_numpy(np.asarray(states))), self.metadata["temperature"]).numpy()
                np.add.at(result, owners, -self.metadata["gamma"] * np.asarray(weights) * values)
        result[absorbing] = 0
        return float(result[0]) if scalar else result

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode())
        for value in self.network.state_dict().values():
            h.update(value.cpu().numpy().tobytes())
        if self.model is not None:
            for pose in sorted(self.model.images):
                h.update(self.model.images[pose].tobytes())
        return h.hexdigest()

    def save(self, path):
        payload = dict(metadata=self.metadata, network=self.network.state_dict(), fingerprint=self.fingerprint())
        if self.model is not None:
            payload["images"] = {str(pose): torch.from_numpy(image.copy()) for pose, image in self.model.images.items()}
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload["metadata"]
        if meta["schema"] != SCHEMA or meta["code_version"] != code_version():
            raise ValueError("Reward implementation/schema changed; reproduce with recorded code version")
        network = mlp(78, 1) if meta["method"] == "airl" else mlp()
        network.load_state_dict(payload["network"])
        result = cls(network, method=meta["method"], representation=meta["domain"],
                     view_size=meta["view_size"], gamma=meta["gamma"], temperature=meta["temperature"])
        result.metadata = meta
        if result.model is not None:
            result.model.images = {}
            for key, image in payload["images"].items():
                pose = tuple(int(v) for v in key.strip("()").split(","))
                array = image.numpy().copy()
                array.setflags(write=False)
                result.model.images[pose] = array
        if result.fingerprint() != payload["fingerprint"]:
            raise ValueError("Reward bundle integrity failure")
        return result
