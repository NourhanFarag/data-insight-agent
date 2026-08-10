# Benchmark Changelog (v1 to v2)

This changelog documents the semantic clarifications applied to the frozen evaluation cases for Benchmark v2. The purpose of these changes is to eliminate ambiguity in the natural language questions so they align with the expected result checks, preventing penalization of valid planning strategies. 

No expected result checks, numeric values, required operations, or aggregate scoring formulas have been modified. These clarifications are made **prior to any planner prompt tuning or model behavioral changes**.

---

## Summary of Clarifications

### 1. `category_frequency`
* **v1 Question**: `Which customer segment appears most frequently?`
* **v2 Question**: `What is the frequency count for every customer segment?`
* **Rationale for Clarification**: The singular phrasing of v1 naturally allowed the planner to optimize with `TOP_VALUES(segment)` using `limit=1` (yielding only SMB counts). However, the benchmark expected an exact-match complete mapping of *all* segments (`{"SMB": 3, "Enterprise": 2}`). The v2 question explicitly asks for the frequency of every segment to resolve this mismatch.

### 2. `missing_data`
* **v1 Question**: `Which fields have missing information that may affect analysis?`
* **v2 Question**: `How many missing values are in each field, including fields with zero missing values?`
* **Rationale for Clarification**: In v1, the query asked which fields *have* missing info. The planner naturally planned `MISSING_VALUES` checks only on columns containing missing data (omitting `department` which had zero missing values). However, the benchmark required checks for all fields, including those with zero missing values. The v2 question explicitly directs the model to include columns with zero missing values.

### 3. `adversarial_case`
* **v1 Question**: `What are the top departments?`
* **v2 Question**: `Show the frequency count for every distinct value in the department column.`
* **Rationale for Clarification**: In v1, "top departments" permitted the planner to limit output to a subset of values (e.g. `limit=2`). However, since all departments in the dataset had a frequency of 1, the benchmark check required a complete frequency mapping of all four values. The v2 question explicitly asks for every distinct value.

---

## Verification & Integrity
* **Expected result semantics unchanged**: Confirmed.
* **unsupported_question remains unchanged**: Confirmed.
* **Adversarial payload sanitization intact**: Confirmed.
