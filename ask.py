import json
import os
import sys
import time
import ollama
from make_index import VectorStore
from yaspin import yaspin
import rich, rich.table, rich.console, rich.style,rich.prompt,rich.pager

console = rich.console.Console()

PROMPT = """
あなたは利用者の役に立つために最善を尽くす助手です。
1. 以下の質問のヒントを読み込んでください。
3. それを使用して、質問に対する返答を行います。
## 質問のヒント
{text}
## 質問
{input}
""".strip()



MAX_PROMPT_SIZE = 1024
RETURN_SIZE = 250

ask_loading = False


def ask(input_str, index_file):
	vs = VectorStore(index_file)
	samples = vs.get_sorted(input_str)
	to_use: list[str] = []
	table = rich.table.Table(
		title="ベクトル検索結果", title_style=rich.style.Style(italic=False),show_lines=True
	)
	table.add_column("ページ名", no_wrap=True)
	table.add_column("類似度", no_wrap=True)
	table.add_column("テキスト", overflow="fold")
	for sim, body, title in samples[0:4]:
		to_use.append(f"### ページ「{title}」の一部(類似度:{round(sim,4)}):\n" + body)
		table.add_row(title, str(round(sim, 4)), body)
	# return
	text = "\n\n".join(to_use)
	prompt = PROMPT.format(input=input_str, text=text)
	console.print(table)
	if not rich.prompt.Confirm.ask("Continue?"):
		return

	global ask_loading
	ask_loading = True
	response = ollama.chat(
        model="hf.co/alfredplpl/llm-jp-3-1.8b-instruct-gguf:IQ4_XS",
		messages=[{"role": "user", "content": prompt}],
		stream=True,
	)
	# show question and answer
	starttime = time.time()
	with yaspin() as sp:
		sp.text = f"LLM is reading prompt({len(prompt)} chars)..."
		sp._timer = True
		for chunk in response:
			if ask_loading:
				sp.ok(
					f"{round(time.time() - starttime,1)} seconds elapsed in reading({len(prompt)} chars)."
				)
				sp._timer = False
				print(f">>>> {input_str}")
				ask_loading = False
			print(chunk.message.content or "", end="", flush=True)
	print(f"\n{round(time.time() - starttime,1)} seconds elapsed")

if __name__ == "__main__":
	ask(" ".join(sys.argv[1:]), "qualia-san.pickle")
