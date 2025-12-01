from flask import Flask, request
from markupsafe import Markup
from string import Template
import html

from minisearch.index import PositionalIndex
from minisearch.search import Searcher      

from .sample_docs import SAMPLE_DOCS

HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IR Demo</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
    .box { max-width: 900px; margin: 0 auto; }
    input[type=text] { width: 100%; padding: 12px 14px; font-size: 16px; border: 1px solid #ccc; border-radius: 10px; }
    .hit { border: 1px solid #eee; border-radius: 12px; padding: 12px 14px; margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .hit h3 { margin: 0 0 6px; font-size: 18px; }
    .hit small { color: #666; }
    mark { background: #fffa87; padding: 0 2px; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f1f5f9; color: #0f172a; font-size: 12px; margin-right: 4px; }
  </style>
</head>
<body>
  <div class="box">
    <h1>Мини-поисковик (учебный)</h1>
    <form method="get" action="/">
      <input autofocus name="q" type="text" placeholder='Например: title:поиск AND "обратный индекс"  или  токен NEAR/3 индекс' value="$q">
    </form>
    <p><small>Операторы: AND, OR, NOT, скобки ( ), фразы в "", NEAR/k, wildcard (* ?), fuzzy ~k.</small></p>
    <hr/>
    $results
  </div>
</body>
</html>
""")

def _render_results(sr: Searcher, q: str) -> str:
    if not q:
        return "<p><i>Введите запрос…</i></p>"
    hits = sr.search(q)
    if not hits:
        return f"<p>Ничего не найдено для <b>{html.escape(q)}</b></p>"
    out = [f"<p><b>{len(hits)}</b> результатов:</p>"]
    for doc_id, score in hits:
        doc = sr.idx.docs[doc_id]
        tags = ''.join(f"<span class='pill'>{html.escape(t.strip())}</span>" for t in doc.get('tags','').split(','))
        snip = sr.make_snippet(doc_id, q)
        out.append(f"""
        <div class="hit">
          <h3>{html.escape(doc.get('title','(без названия)'))}</h3>
          <small>score: {score:.4f}</small>
          <div style="margin-top:6px">{snip}</div>
          <div style="margin-top:8px">{tags}</div>
        </div>
        """)
    return "\n".join(out)

def create_app():
    idx = PositionalIndex()
    for i, d in enumerate(SAMPLE_DOCS):
        idx.add_document(i, d)
    idx.commit()
    sr = Searcher(idx)
    app = Flask(__name__)

    @app.get("/")
    def home():
        q = request.args.get("q", "").strip()
        res = _render_results(sr, q)
        return HTML_TEMPLATE.substitute(q=html.escape(q), results=Markup(res))

    return app

def main():
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)

if __name__ == "__main__":
    main()
