# Agent Skills

This directory holds agent skills that belong to Frontier's development
workflow. Each skill is a directory with a `SKILL.md` at its top; an agent
loads `SKILL.md` and reads the skill's other files when the skill directs it
to.

| Skill | Use |
| --- | --- |
| `frontier-calibration/` | Compare Frontier with vLLM-BS ground truth: collect a case, gate the E2E metrics, find error sources, and apply decided corrections. |

## Installing a Skill

Agents discover skills in their own skill directory. Link the repository copy
there so the agent always reads the checked-in version:

```bash
ln -s "$PWD/skills/frontier-calibration" ~/.claude/skills/frontier-calibration
```

Use the skill directory of your agent in place of `~/.claude/skills/`.

## Host Bindings

The checked-in skills name no host paths, accounts, or platforms. A skill that
needs them reads `site.local.md` beside its `SKILL.md`, which `.gitignore`
keeps out of the repository. Write it per machine; for `frontier-calibration`
it names the vLLM-BS checkout, the calibration artifacts root, the scratch
root (`FRONTIER_TMP_ROOT`), the GPU platform and device generation, and the
Python command.

## Checking a Skill

`frontier-calibration` carries a structural check of its entry catalog, call
graph, repository paths, and host-neutral wording:

```bash
python3 skills/frontier-calibration/tests/pressure_scenarios.py skills/frontier-calibration
```

It prints `v2_pressure=PASS` on success.
