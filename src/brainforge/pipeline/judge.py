def teacher_signal(data: dict) -> bool | None:
    if "vulnerability_found" in data:
        return bool(data["vulnerability_found"])
    if "issues" in data:
        return bool(data["issues"])
    return None


def is_unresolved(judge_data: dict) -> bool:
    return judge_data.get("verdict") == "insufficient_information"


def compute_agreement(judge_data: dict, teacher_results: dict) -> dict[str, bool]:
    judge_positive = judge_data.get("verdict") == "confirmed"
    agreement: dict[str, bool] = {}
    for role, data in teacher_results.items():
        signal = teacher_signal(data)
        if signal is not None:
            agreement[role] = signal == judge_positive
    return agreement
