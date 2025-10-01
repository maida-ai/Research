import re
import subprocess


def load_md(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()


def update_gh_issue(
    issue_number: int,
    new_title: str | None = None,
    new_body: str | None = None,
):
    if new_title is None and new_body is None:
        raise ValueError("Nothing to update")
    args = []
    if new_title is not None:
        args.append(f"--title={new_title}")
    if new_body is not None:
        args.append(f"--body={new_body}")
    args.append(str(issue_number))
    subprocess.run(["gh", "issue", "edit"] + args)


def parse_md(
    md_content: str,
    re_start_token: re.Pattern,
    re_end_token: re.Pattern,
) -> list[dict]:
    issues = {}  # {issue_number: {title: str, body: str}}
    current_issue = None  # number
    current_title = None
    current_body = []
    for line in md_content.split("\n"):
        if current_issue is not None:
            if re_end_token.match(line):
                issues[current_issue] = {
                    "title": current_title,
                    "body": "\n".join(current_body).strip(),
                }
                current_issue = None
                current_title = None
                current_body = []
            else:
                current_body.append(line)
        elif re_start_token.match(line):
            _, current_issue, current_title = line.split(" ", 2)
            current_issue = int(current_issue)
            current_title = current_title.strip()
            current_body = []

    return issues



def main():
    md_content = load_md("plans.md")
    issues = parse_md(md_content, re.compile(r"# \d+"), re.compile(r"---"))
    for key, value in issues.items():
        print(f"Updating issue {key}...")
        update_gh_issue(key, value["title"], value["body"])


if __name__ == "__main__":
    main()
