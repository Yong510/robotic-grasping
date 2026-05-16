import json


CLASS_NAMES_DEFAULT = {
    0: "standard_part",
    1: "package",
    2: "abnormal_part",
}

RISK_NAMES_DEFAULT = {
    0: "low",
    1: "medium",
    2: "high",
}


def build_robot_decision_prompt(user_command, objects):
    """
    Build a strict-JSON prompt for an LLM decision layer.

    Args:
        user_command: natural-language task instruction.
        objects: list of object perception results.
    """
    payload = {
        "role": "safety_aware_robot_grasp_decision",
        "user_command": user_command,
        "objects": objects,
        "rules": [
            "If risk is high, do not grasp automatically.",
            "If grasp_score is lower than 0.60, do not grasp.",
            "If the object class does not match the user command, do not select it.",
            "Return strict JSON only.",
        ],
        "output_schema": {
            "selected_object_id": "int or null",
            "action": "grasp | do_not_grasp | request_human_review",
            "target": "target place or null",
            "reason": "short explanation",
        },
    }
    return (
        "You are a safety-aware robot grasp decision model. "
        "Read the following JSON and output strict JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def rule_based_decision(user_command, objects):
    """
    Safety fallback when the LLM is unavailable or its JSON output is invalid.
    """
    candidates = []
    cmd = user_command.lower()

    for obj in objects:
        risk = obj.get("risk", "unknown")
        cls = obj.get("class", "unknown")
        score = float(obj.get("grasp_score", 0.0))

        if risk == "high":
            continue
        if score < 0.60:
            continue

        if "异常" in user_command or "abnormal" in cmd:
            if cls == "abnormal_part":
                return {
                    "selected_object_id": obj.get("id"),
                    "action": "request_human_review",
                    "target": None,
                    "reason": "目标为异常件，建议人工复核后再执行。",
                }

        if "标准" in user_command or "standard" in cmd:
            if cls != "standard_part":
                continue

        if "包装" in user_command or "package" in cmd:
            if cls != "package":
                continue

        candidates.append(obj)

    if len(candidates) == 0:
        return {
            "selected_object_id": None,
            "action": "do_not_grasp",
            "target": None,
            "reason": "没有满足任务要求且风险可接受的目标。",
        }

    best = max(candidates, key=lambda o: float(o.get("grasp_score", 0.0)))
    return {
        "selected_object_id": best.get("id"),
        "action": "grasp",
        "target": best.get("target", None),
        "reason": "选择风险可接受且抓取置信度最高的目标。",
    }


if __name__ == "__main__":
    demo_objects = [
        {
            "id": 0,
            "class": "standard_part",
            "risk": "low",
            "grasp_score": 0.93,
            "x": 120,
            "y": 180,
            "angle": 0.35,
            "width": 48.0,
        },
        {
            "id": 1,
            "class": "abnormal_part",
            "risk": "high",
            "grasp_score": 0.91,
            "x": 210,
            "y": 160,
            "angle": -0.22,
            "width": 52.0,
        },
    ]
    command = "抓取低风险标准零件，不要抓异常件"
    print(build_robot_decision_prompt(command, demo_objects))
    print(json.dumps(rule_based_decision(command, demo_objects), ensure_ascii=False, indent=2))
