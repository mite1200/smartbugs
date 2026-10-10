from typing import TYPE_CHECKING, Any, Optional

import sb.docker
import sb.errors

if TYPE_CHECKING:
    import sb.tasks


ExecutionResult = tuple[Optional[int], list[str], Optional[bytes], list[str], dict[str, Any]]


def execute(task: "sb.tasks.Task") -> ExecutionResult:
    runner = getattr(task.tool, "runner", "docker")
    if runner == "docker":
        return sb.docker.execute(task)
    if runner == "llm":
        from sb.llm.runner import execute as execute_llm

        return execute_llm(task)
    raise sb.errors.SmartBugsError(f"Unknown tool runner: {runner}")
