"""Guards the local fleet config against reintroducing the degraded-replica penalty.

The backend image adds DEGRADED_MS to every request served by DEGRADED_REPLICA
("3" by default), so every replica must pin DEGRADED_MS to 0.
"""
import re
from pathlib import Path

_COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.yml"


def _service_blocks() -> dict[str, str]:
    text = _COMPOSE.read_text()
    parts = re.split(r"^  (llm-service-\d+):\s*$", text, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def test_fleet_has_three_replicas():
    assert sorted(_service_blocks()) == ["llm-service-1", "llm-service-2", "llm-service-3"]


def test_no_replica_is_degraded():
    for name, block in _service_blocks().items():
        assert re.search(r'^\s+DEGRADED_MS:\s*"0"\s*$', block, flags=re.M), name
