# Contributing

Thank you for being interested in contributing to Tachyon!

Please feel welcome to dive in. Read the following sections to learn about how to ask questions and how to work on something. Whether you're looking to write code, contribute to documentation, report bugs, or just ask questions, we appreciate all forms of contribution.

All members of our community are expected to follow our [Code of Conduct](./CODE_OF_CONDUCT.md). In all our interactions, let's make sure to create a welcoming and friendly space for everyone.

We're really glad you're reading this, because we need volunteer developers to help this project come to fruition.

## Issues

The best way to contribute to our projects is by opening a [new issue](https://github.com/kroma-network/tachyon/issues/new/choose) or tackling one of the issues listed [here](https://github.com/kroma-network/tachyon/issues).

## CI (Github Actions)

We use GitHub Actions to verify that the code in your PR passes all our checks.

When you submit your PR (or later change that code), a CI build will automatically be kicked off. A note will be added to the PR, and will indicate the current status of the build.

## Coding Style

All of our code follows the [Google Style Guides](https://google.github.io/styleguide/). We use [cpplint](https://github.com/cpplint/cpplint) and [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to keep our code clean and readable. If you want to make sure your code passes our checks and follows our rules, follow the guide below.

### Formatter: [clang-format](https://clang.llvm.org/docs/ClangFormat.html)(version 15)

Please install clang-format version 15 and run it before committing your changes.
Ensure you use version 15 to prevent conflicts.

For NPM:

```shell
> npm install -g clang-format
```

For MacOS:

```shell
> brew install clang-format
```

For Ubuntu - Version 14 is a default in Ubuntu, so you will need to upgrade to version 15:

```shell
> sudo apt install clang-format clang-format-15
> sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-15 100 --slave /usr/share/man/man1/clang-format.1.gz clang-format.1.gz /usr/share/man/man1/clang-format-15.1.gz
[sudo] password for <user>:
update-alternatives: using /usr/bin/clang-format-15 to provide /usr/bin/clang-format (clang-format) in auto mode
> sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-14 20 --slave /usr/share/man/man1/clang-format.1.gz clang-format.1.gz /usr/share/man/man1/clang-format-14.1.gz
> sudo update-alternatives --config clang-format
```

**Run clang-format:**

```shell
> find . -name '*.cc' -or -name '*.h' | xargs clang-format -style=file -i
```

### Linter: [cpplint](https://github.com/cpplint/cpplint)

cpplint is a tool for checking C++ code against Google's style guide. We've configured it to ignore certain warnings(`legal/copyright`, `whitespace/line_length`, `build/namespaces`, `runtime/references`) for our project.

Before committing, ensure your changes pass the cpplint checks.

**Install cpplint:**

```shell
> pip install cpplint
```

**Run cpplint:**

```shell
> find . -iname "*.h" -o -iname "*.cc" | grep -v -e "^./tachyon/base/" -e "^./tachyon/device/" | xargs cpplint --filter=-legal/copyright,-whitespace/line_length,-build/namespaces,-runtime/references
```

## Steps to commit

1. Leave the issues. (If the issue is a straightforward one, it is indeed possible to skip this process.)
2. Create a new branch whose name starts with an imperative verb above like `feat/implement-xyz`.
   Branch name should consist of [a-z|0-9|-]. The prefix keyword should be one of followings defined [Commit type](#commit-type)
3. Make your changes.
4. Run a formatting tool if you make changes to codes.
5. Run a lint tool if you make changes to codes.
6. Run unittests if you make changes to codes.
7. Rebase your local repository.

   ```shell
   > git fetch origin -p
   > git rebase origin/dev
   ```

## What to check

Before Pull Request, you should check the following first.

- Whether there is typo in the subject and body of your commit.
- Whether the subject of your commit explains well about your changes.
- Whether your commits are well split in semantics.
- Whether your PR contains any intermediate changes among commits.

## How to check your changes

You can check above with `git` commands. Follow this guideline if you encounter any of the situations below.
You can check your status by `git status`.

- Changes not staged for commit: check with `git diff`.
- Changes to be committed: check with `git diff --cached`.
- Committed: check with `git log -p`.

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

### Commit Type

Must be one of the following:

- **build**: Changes that affect the build system or external dependencies.
- **chore**: Fixes typo; no production code change.
- **ci**: Changes to our CI configuration files and scripts.
- **docs**: Documentation only changes.
- **feat**: A feature addition or removal.
- **fix**: A bug fix. Just a fixing typo must be typed as a `chore`.
- **perf**: A code change that improves performance.
- **refac**: A code change that improves readability or code structure.
  This may incur internal features. Also, this may increase performance, but it's different
  from `perf` type in that performance improvement is not the goal.
- **test**: Adding missing tests or correcting existing tests.

**NOTE:** Some repositories such as `.github` itself only contain documents. So it may be
redundant to add commit type. In this case, the commit type may be omitted, and the branching rule is also relaxed.

### Commit Scope

Exceptionally, commit type `docs` may omit scope. e.g, docs: update contracts/README.md.
If the change affects more than one scope, the commit scope may be omitted.
e.g, feat: add Colosseum contract

## How to merge

If a PR author receives `Comment` from reviewers, not `Request changes`, it is recommended
to wait for the comment authors to confirm the changes that the PR author made after the review.
Therefore, once the PR author has reflected all the requested changes, please re-request
for the comment authors to review again so that they can review and `Approve` the changes.
The PR author should merge and close the PR after receiving `Approve` from all comment authors.
