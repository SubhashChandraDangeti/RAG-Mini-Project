# Git commands for this project

Run commands from the project folder in the VS Code PowerShell terminal.

## First upload

```powershell
git init -b main
git remote add origin https://github.com/SubhashChandraDangeti/RAG-Mini-Project.git
git status --short
git add .gitignore README.md REPORT.md requirements.txt rag_baseline_vs_reranking.ipynb GIT_GUIDE.md
git diff --cached --stat
git commit -m "Add chess RAG experiment and report draft"
git push -u origin main
```

| Command | Purpose |
| --- | --- |
| `git init -b main` | Create local Git history with `main` as the branch name. Run once. |
| `git remote add origin ...` | Name the GitHub repository `origin`. Run once. |
| `git status --short` | See files that are new, modified, or staged. |
| `git add ...` | Stage the named files for the next commit. |
| `git diff --cached --stat` | Review a summary of the staged changes. |
| `git commit -m "..."` | Record the staged changes in local history. |
| `git push -u origin main` | Upload commits and connect the local branch to `origin/main`. |

The `.gitignore` file excludes credentials, Python environments, checkpoints, and local PDFs. Only explicitly staged files enter the commit. A commit saves history locally; a push sends that history to GitHub.

## Later updates

For example, after completing your report:

```powershell
git status --short
git diff -- REPORT.md
git add REPORT.md
git commit -m "Complete experimental analysis in report"
git push
```

After saving new notebook outputs, stage `rag_baseline_vs_reranking.ipynb` as well. `git push` can be used without the remote and branch names after the initial `-u` command.

## Useful checks

```powershell
git log --oneline -5
git remote -v
git status
```

These show recent commits, the configured GitHub URL, and whether any local changes remain uncommitted.
