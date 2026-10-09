from sb.llm.profiles.base import AnalysisProfile
from sb.llm.profiles.reentrancy import ReentrancyProfile

PROFILES: dict[str, type[AnalysisProfile]] = {"reentrancy": ReentrancyProfile}


def load_profile(profile_id: str) -> AnalysisProfile:
    try:
        return PROFILES[profile_id]()
    except KeyError as error:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown analysis profile {profile_id!r}; available profiles: {available}"
        ) from error
