import time
import json
import pickle
import ollama
import numpy as np
from tqdm import tqdm
import dotenv
import os
import typing
from yaspin import yaspin



BLOCK_SIZE = 512
EMBED_MAX_SIZE = 4096

dotenv.load_dotenv()


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
	v1_norm = np.linalg.norm(v1)
	v2_norm = np.linalg.norm(v2)
	if v1_norm == 0 or v2_norm == 0:
		raise ZeroDivisionError("one of the norms is 0")
	v1_v2_dot = np.dot(v1, v2)
	sim = v1_v2_dot / (v1_norm * v2_norm)
	return sim


def embed_text(text: str, queryortext: typing.Literal["query", "text"]):
	"take text, return embedding vector"
	text = text.replace("\n", " ")
	if queryortext == "text":
		text = "文章: " + text
	elif queryortext == "query":
		text = "クエリ: " + text
	res = ollama.embed(input=[text], model="kun432/cl-nagoya-ruri-large")
	return res.embeddings


def update_from_scrapbox(json_file: str, out_index: str, in_index=None):
	"""
	out_index: Output index file name
	json_file: Input JSON file name (from cosense)
	in_index: Optional input index file name. It is not modified and is used as cache to reduce API calls.
	out_index: 出力インデックスファイル名
	json_file: 入力JSONファイル名 (cosenseからの)
	in_index: オプショナルな入力インデックスファイル名。変更されず、APIコールを減らすためのキャッシュとして使用されます。

	# usage
	## create new index
	update_from_scrapbox(
		"from_scrapbox/nishio.json",
		"nishio.pickle")

	## update index
	update_from_scrapbox(
		"from_scrapbox/nishio-0314.json", "nishio-0314.pickle", "nishio-0310.pickle")
	"""
	if in_index is not None:
		cache = pickle.load(open(in_index, "rb"))
	else:
		cache = None

	vs = VectorStore(out_index)
	data = json.load(open(json_file, encoding="utf8"))
	print("")
	records: list[tuple[str, str]] = []
	for p in data["pages"]:
		title = p["title"]
		body = " ".join(p["lines"])
		while body:
			records.append((body[0:500], title))
			print(f"\x1b[2K\x1b[0Gtotal {len(records)} indexs", end="")
			if len(body) <= 500:
				break
			body = body[384:]
	start_time = time.time()
	print(f"\x1b[2K\x1b[0Gtotal {len(records)} indexs")
	for index, body in enumerate(records):
		vs.add_record(*body, cache)
		elapsed_time = time.time() - start_time
		remain_time = (len(records) - index - 1) * (elapsed_time / (index + 1))
		print(
			f"\x1b[2K\x1b[0G{index+1:0{len(str(len(records)))}}/{len(records)}, {round(elapsed_time/60):02}:{round(elapsed_time,1)%60:04.1f}/{round(remain_time/60):02}:{round(remain_time,1)%60:04.1f}, @{round(elapsed_time/(index+1),1)}s:{body[1]}({len(body[0])})",
			end="",
			flush=True,
		)
	vs.save()


class VectorStore:
	def __init__(self, name, create_if_not_exist=True):
		self.name = name
		try:
			pic = open(self.name, "rb")
			self.cache = pickle.load(pic)
			pic.close()
		except FileNotFoundError as e:
			if create_if_not_exist:
				self.cache: dict[
					str, tuple[typing.Sequence[typing.Sequence[float]], str]
				] = {}
			else:
				raise

	def add_record(self, body, title, cache=None):
		try:
			if cache is None:
				cache = self.cache
			if body not in cache:
				# call embedding API

				self.cache[body] = (embed_text(body, "text"), title)

			elif body not in self.cache:
				# in cache and not in self.cache: use cached item
				self.cache[body] = cache[body]

			return self.cache[body]
		except Exception as e:
			print("error:", e)

	def get_sorted(self, query):
		with yaspin(text="Embedding query...") as sp:
			q = np.array(embed_text(query, "query")[0])
		buf = []
		for body, (v, title) in tqdm(self.cache.items()):
			buf.append((cos_sim(q, np.array(v[0])), body, title))
			# q.dot(v)
		buf.sort(reverse=True)
		return buf

	def save(self):
		pickle.dump(self.cache, open(self.name, "wb"))


if __name__ == "__main__":
	# Sample default arguments for update_from_scrapbox()
	JSON_FILE = "from_scrapbox/qualia-san.json"
	INDEX_FILE = "qualia-san.pickle"

	update_from_scrapbox(JSON_FILE, INDEX_FILE)
