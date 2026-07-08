"""Demo: synthetic tool observation preserves call/result pairing."""

from fairlib.core.tool_observations import synthetic_tool_observation


def main() -> None:
    obs = synthetic_tool_observation("web_search", "denied by security policy")
    print(obs)
    assert "web_search" in obs
    print("Synthetic tool result demo complete.")


if __name__ == "__main__":
    main()
