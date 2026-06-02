# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Commit Hygiene

**Commit messages must be structured, specific, and reviewable.**

When creating commits:
- Use a standard subject line format: `type(scope): short summary`
- Prefer conventional types such as `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Write the subject line in Chinese
- Keep the first line focused on the main change, not a vague summary like "update files"
- Add multiple `-m` flags to describe each distinct change you made
- Each extra `-m` message should describe one concrete item in Chinese, preferably starting with `- `

Example:
```bash
git commit \
  -m "fix(io): 规范化 xas 加载器元数据" \
  -m "- 移除 CLAESS 读取器中错误的能量单位回退逻辑" \
  -m "- 补充缺失元数据字段的回归测试" \
  -m "- 更新文档示例以匹配新的加载结果"
```

The test:
- A reviewer should understand what changed from the commit message alone
- Distinct changes in one commit should be visible as distinct `-m` entries
- The subject line and each extra `-m` entry should be written in Chinese

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
