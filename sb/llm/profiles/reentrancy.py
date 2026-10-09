from copy import deepcopy
from typing import Any

from sb.llm.models import ParseResult, PreprocessingConfig
from sb.llm.profiles.base import AnalysisProfile

REENTRANCY_DEFINITION = """A smart contract is vulnerable to reentrancy if an external interaction can pass
control to potentially attacker-controlled code, enabling the contract to be
re-entered before the initial invocation has finished, and this nested execution
can cause a security-relevant negative effect or contract state that would not
occur under a corresponding non-reentrant execution.

The mere existence of an external interaction, a possible callback opportunity,
or a state change after an external interaction is not sufficient for a vulnerable
classification. A contract should be classified as safe if potential re-entry is
effectively prevented or if it cannot lead to a security-relevant semantic
difference."""

STRUCTURED_REASONING = """Before your final classification, perform the following structured analysis
steps:

1. Identify all external interactions

Identify any point where the contract execution passes control outside of the
contract.

2. Identify relevant state

Determine which contract state is read or changed before and after every external
interaction and which state is relevant to the specific operation.

3. Analyze potential re-entry

Check if control can be passed back to the contract before the initial invocation
has finished. Take into account both re-entry into the same function and re-entry
into other functions which may interact with relevant shared state.

4. Analyze protection measures

Determine if reentrancy is effectively prevented or rendered harmless through
the contract's logic, e.g., by finishing state updates before the interaction or
through other protection mechanisms such as reentrancy guards.

5. Evaluate semantic effects

If re-entry is possible, determine whether the nested execution can produce a
security-relevant negative effect or contract state that would not occur under a
corresponding non-reentrant execution. Do not classify a contract as reentrant
only because a suspicious code pattern exists.

6. Make the final verdict

Classify a contract as reentrant only if the preceding analysis supports a concrete
harmful reentrant execution path. Otherwise, classify the contract as safe."""

DEMONSTRATIONS = """Example 1

Solidity contract:

001 | pragma solidity ^0.8.20;
002 |
003 | contract DemoSameFunction {
004 |     mapping(address => uint256) private balances;
005 |
006 |     function deposit() external payable {
007 |         balances[msg.sender] += msg.value;
008 |     }
009 |
010 |     function withdraw() external {
011 |         uint256 amount = balances[msg.sender];
012 |         require(amount > 0);
013 |         (bool success,) = msg.sender.call{value: amount}("");
014 |         require(success);
015 |         balances[msg.sender] = 0;
016 |     }
017 | }

Expected output:

{
  "classification": "Reentrant",
  "vulnerable_functions": ["withdraw"],
  "reentered_functions": ["withdraw"],
  "reentrancy_type": "same_function",
  "relevant_lines": [
    {"line": 13, "role": "external_interaction"},
    {"line": 15, "role": "state_effect"},
    {"line": 10, "role": "reentry_target"}
  ],
  "explanation": "The external interaction occurs before the caller's balance is cleared. The caller can re-enter withdraw while the original balance is still available and trigger repeated withdrawals."
}

Example 2

Solidity contract:

001 | pragma solidity ^0.8.20;
002 |
003 | contract DemoCEI {
004 |     mapping(address => uint256) private balances;
005 |
006 |     function deposit() external payable {
007 |         balances[msg.sender] += msg.value;
008 |     }
009 |
010 |     function withdraw() external {
011 |         uint256 amount = balances[msg.sender];
012 |         require(amount > 0);
013 |         balances[msg.sender] = 0;
014 |         (bool success,) = msg.sender.call{value: amount}("");
015 |         require(success);
016 |     }
017 | }

Expected output:

{
  "classification": "Safe",
  "vulnerable_functions": [],
  "reentered_functions": [],
  "reentrancy_type": null,
  "relevant_lines": [
    {"line": 13, "role": "mitigation"},
    {"line": 14, "role": "external_interaction"}
  ],
  "explanation": "The caller's balance is cleared before control is transferred externally, so re-entry cannot reuse the original balance."
}

Example 3

Solidity contract:

001 | pragma solidity ^0.8.20;
002 |
003 | contract DemoCrossFunction {
004 |     mapping(address => uint256) private balances;
005 |     bool private withdrawLocked;
006 |
007 |     modifier noRecursiveWithdraw() {
008 |         require(!withdrawLocked);
009 |         withdrawLocked = true;
010 |         _;
011 |         withdrawLocked = false;
012 |     }
013 |
014 |     function deposit() external payable {
015 |         balances[msg.sender] += msg.value;
016 |     }
017 |
018 |     function withdraw() external noRecursiveWithdraw {
019 |         uint256 amount = balances[msg.sender];
020 |         require(amount > 0);
021 |         (bool success,) = msg.sender.call{value: amount}("");
022 |         require(success);
023 |         balances[msg.sender] = 0;
024 |     }
025 |
026 |     function transferBalance(address recipient, uint256 amount) external {
027 |         require(balances[msg.sender] >= amount);
028 |         balances[msg.sender] -= amount;
029 |         balances[recipient] += amount;
030 |     }
031 | }

Expected output:

{
  "classification": "Reentrant",
  "vulnerable_functions": ["withdraw", "transferBalance"],
  "reentered_functions": ["transferBalance"],
  "reentrancy_type": "cross_function",
  "relevant_lines": [
    {"line": 21, "role": "external_interaction"},
    {"line": 23, "role": "state_effect"},
    {"line": 26, "role": "reentry_target"},
    {"line": 28, "role": "state_effect"},
    {"line": 29, "role": "state_effect"}
  ],
  "explanation": "The guard prevents recursive calls to withdraw but does not protect transferBalance. During the external interaction, the caller can re-enter transferBalance and move the still-available balance to another address before withdraw clears the original balance."
}

Example 4

Solidity contract:

001 | pragma solidity ^0.8.20;
002 |
003 | contract DemoGuard {
004 |     mapping(address => uint256) private balances;
005 |     bool private locked;
006 |
007 |     modifier nonReentrant() {
008 |         require(!locked);
009 |         locked = true;
010 |         _;
011 |         locked = false;
012 |     }
013 |
014 |     function deposit() external payable nonReentrant {
015 |         balances[msg.sender] += msg.value;
016 |     }
017 |
018 |     function withdraw() external nonReentrant {
019 |         uint256 amount = balances[msg.sender];
020 |         require(amount > 0);
021 |         (bool success,) = msg.sender.call{value: amount}("");
022 |         require(success);
023 |         balances[msg.sender] = 0;
024 |     }
025 | }

Expected output:

{
  "classification": "Safe",
  "vulnerable_functions": [],
  "reentered_functions": [],
  "reentrancy_type": null,
  "relevant_lines": [
    {"line": 18, "role": "mitigation"},
    {"line": 21, "role": "external_interaction"},
    {"line": 23, "role": "state_effect"}
  ],
  "explanation": "Although the external interaction occurs before the balance is cleared, the shared reentrancy guard prevents nested entry into the contract's relevant functions while withdraw is executing."
}"""

COMMON_OUTPUT = """Return only a single valid JSON object with the required common fields:
"classification", "vulnerable_functions", "reentered_functions",
"reentrancy_type", "relevant_lines", and "explanation".

The value of "classification" must be either "Reentrant" or "Safe".

The value of "reentrancy_type" must be one of:
"same_function", "cross_function", "delegated", "create_based", "other", or null.

The value of "role" for entries in "relevant_lines" must be one of:
"external_interaction", "state_effect", "reentry_target", "mitigation", or
"other".

For a safe contract, "vulnerable_functions" and "reentered_functions" must be
empty arrays and "reentrancy_type" must be null.

"vulnerable_functions" means all contract functions that materially participate
in the identified reentrant exploit path.

"reentered_functions" means functions entered during the nested/reentrant
execution.

A safe contract may still contain "relevant_lines" entries, for example lines
containing a mitigation.

The line numbers in the output must refer to the numbered source code below.

The explanation should briefly summarize the evidence for the final
classification and must not be a full internal chain-of-thought transcript."""

REENTRANCY_TYPES = {"same_function", "cross_function", "delegated", "create_based", "other"}
LINE_ROLES = {"external_interaction", "state_effect", "reentry_target", "mitigation", "other"}


class ReentrancyProfile(AnalysisProfile):
    id = "reentrancy"
    version = 1
    task = "vulnerability_detection"
    task_name = "reentrancy"
    primary_classification_field = "classification"
    accepted_labels = ("Reentrant", "Safe")
    preprocessing = PreprocessingConfig(remove_comments=True, add_line_numbers=True)
    prompt_versions = {
        "zero_shot": 1,
        "structured_reasoning": 1,
        "few_shot": 1,
        "structured_reasoning_few_shot": 1,
    }

    def render_prompt(self, strategy: str, source: str) -> str:
        if strategy not in self.prompt_versions:
            raise ValueError(f"unsupported prompt strategy {strategy!r}")
        sections = [
            "You are a security expert for Solidity smart contracts.",
            f"Use the following definition of reentrancy:\n\n{REENTRANCY_DEFINITION}",
            'Analyze the following Solidity contract and classify it as either "Reentrant" or "Safe".',
        ]
        if strategy in {"structured_reasoning", "structured_reasoning_few_shot"}:
            sections.append(STRUCTURED_REASONING)
        sections.append(COMMON_OUTPUT)
        if strategy in {"few_shot", "structured_reasoning_few_shot"}:
            sections.append(DEMONSTRATIONS)
        sections.append(f"Solidity contract:\n\n{source}")
        return "\n\n".join(sections)

    def validate_auxiliary(
        self, parsed: dict[str, Any], source_line_count: int
    ) -> tuple[dict[str, Any], list[str]]:
        result = deepcopy(parsed)
        warnings: list[str] = []
        for field in ("vulnerable_functions", "reentered_functions"):
            value = result.get(field)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                warnings.append(f"{field} must be an array of strings")
        reentrancy_type = result.get("reentrancy_type")
        if reentrancy_type is not None and reentrancy_type not in REENTRANCY_TYPES:
            warnings.append("reentrancy_type has an unexpected value")
        relevant_lines = result.get("relevant_lines")
        valid_lines: list[dict[str, Any]] = []
        if not isinstance(relevant_lines, list):
            warnings.append("relevant_lines must be an array")
        else:
            for index, entry in enumerate(relevant_lines):
                if not isinstance(entry, dict):
                    warnings.append(f"relevant_lines[{index}] must be an object")
                    continue
                line, role = entry.get("line"), entry.get("role")
                if not isinstance(line, int) or isinstance(line, bool):
                    warnings.append(f"relevant_lines[{index}].line must be an integer")
                    continue
                if line < 1 or line > source_line_count:
                    warnings.append(f"relevant_lines[{index}].line is outside the source")
                    continue
                if role not in LINE_ROLES:
                    warnings.append(f"relevant_lines[{index}].role has an unexpected value")
                    continue
                valid_lines.append({"line": line, "role": role})
        result["relevant_lines"] = valid_lines
        if not isinstance(result.get("explanation"), str):
            warnings.append("explanation must be a string")
        if result.get("classification") == "Safe":
            if result.get("vulnerable_functions") != []:
                warnings.append("Safe classification requires empty vulnerable_functions")
            if result.get("reentered_functions") != []:
                warnings.append("Safe classification requires empty reentered_functions")
            if result.get("reentrancy_type") is not None:
                warnings.append("Safe classification requires null reentrancy_type")
        return result, warnings

    def smartbugs_findings(self, result: ParseResult) -> list[dict[str, Any]]:
        if not result.classification_valid or result.classification != "Reentrant":
            return []
        validated = result.validated_response or {}
        finding: dict[str, Any] = {
            "name": "reentrancy",
            "message": validated.get("explanation", "LLM classified the contract as Reentrant."),
            "severity": "high",
        }
        functions = validated.get("vulnerable_functions")
        if isinstance(functions, list) and functions:
            finding["function"] = ", ".join(functions)
        lines = validated.get("relevant_lines") or []
        for preferred_role in ("external_interaction", "reentry_target", "state_effect"):
            selected = next((entry for entry in lines if entry.get("role") == preferred_role), None)
            if selected:
                finding["line"] = selected["line"]
                break
        else:
            if lines:
                finding["line"] = lines[0]["line"]
        finding["classification_valid"] = result.classification_valid
        finding["schema_valid"] = result.schema_valid
        finding["reentrancy_type"] = validated.get("reentrancy_type")
        return [finding]
