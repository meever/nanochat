---
name: buildbook
description: Use this skill when asked to study a codebase and build a polished HTML learning book in learn/book.html
---

When this skill is requested, first read and follow the canonical prompt in:

- `.github/prompts/buildbook.prompt.md`

Execution requirements:

1. Follow the phases exactly (study, plan, chapter writing, build script, HTML design, verification).
2. Create or update chapter markdown files under `learn/chapters/` using `NN-slug-name.md` naming.
3. Build with `learn/build-book.ps1` and generate `learn/book.html`.
4. Ensure KaTeX and Mermaid rendering work correctly and that no raw `$$` math remains unrendered.
5. Keep chapter content concrete and tied to real files/functions in the codebase.

If there are conflicts between ad-hoc instructions and the canonical prompt, prioritize user instructions first, then apply the canonical prompt as the default behavior.