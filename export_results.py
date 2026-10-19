"""Export recorded notebook results without running retrieval or calling an API.

Usage: python export_results.py
Tables use the precision preserved in Jupyter's saved display outputs.
"""
from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'rag_baseline_vs_reranking.ipynb'
RESULTS = ROOT / 'results'


class TableParser(HTMLParser):
    """Read the first HTML table serialized by a notebook DataFrame output."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.parts = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        if tag in ('td', 'th'):
            self.in_cell = True
            self.parts = []

    def handle_data(self, text):
        if self.in_cell:
            self.parts.append(text)

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self.row.append(' '.join(''.join(self.parts).split()))
            self.in_cell = False
        if tag == 'tr' and self.row:
            self.rows.append(self.row)


def dataframe_rows(data):
    parser = TableParser()
    parser.feed(''.join(data['text/html']))
    headers = parser.rows[0]
    # Pandas writes an unnamed index column before the actual columns.
    return [dict(zip(headers[1:], row[1:])) for row in parser.rows[1:]]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric_rank(value):
    return None if value == 'NaN' else int(float(value))


def main():
    notebook_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    ranks = latency = means = None
    comparisons = []
    answers = []
    original_plot = None
    tests = []

    for cell_index, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and source.startswith('STRESS_TESTS ='):
            tests = ast.literal_eval(ast.parse(source).body[0].value)
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            plain = ''.join(data.get('text/plain', []))
            if 'text/html' in data:
                rows = dataframe_rows(data)
                fields = rows[0].keys()
                if 'baseline_evidence_rank' in fields:
                    ranks = rows
                elif 'baseline_rank' in fields and 'cross_score' in fields:
                    comparisons.append(rows)
                elif 'baseline_retrieval_ms' in fields:
                    if 'test_id' in fields:
                        latency = rows
                    else:
                        means = rows[0]
            markdown = ''.join(data.get('text/markdown', []))
            if markdown.startswith('### Q'):
                answers.append(markdown)
            if 'image/png' in data and 'profile_query' in source:
                original_plot = ''.join(data['image/png'])

    assert ranks and latency and means, 'Save the completed tables in the notebook first.'
    assert len(comparisons) == len(tests), 'Expected one comparison table per query.'
    assert original_plot, 'The saved notebook latency graph was not found.'

    candidate_tables = {test['test_id']: rows for test, rows in zip(tests, comparisons)}
    qualifying = []
    for row in ranks:
        old = numeric_rank(row['baseline_evidence_rank'])
        new = numeric_rank(row['reranked_evidence_rank'])
        if old is not None and new is not None and 4 <= old <= 10 and new <= 3:
            evidence = [r for r in candidate_tables[row['test_id']]
                        if r['is_reference_evidence'] == 'True']
            evidence.sort(key=lambda r: int(r['reranked_rank']))
            qualifying.append({'query': row['test_id'], 'chunk_id': evidence[0]['chunk_id'],
                               'page': evidence[0]['page'], 'baseline_rank': old,
                               'reranked_rank': new})
    assert len(qualifying) >= 2, 'At least two recorded qualifying cases are needed.'
    assert all(any(a.startswith('### ' + r['query'] + ':') for a in answers)
               for r in qualifying), 'Recorded answer pairs are missing.'

    tables = RESULTS / 'tables'
    graphs = RESULTS / 'graphs'
    tables.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)

    write_csv(tables / 'rank_results.csv', ranks)
    write_csv(tables / 'rank_shifts.csv', qualifying)
    write_csv(tables / 'latency.csv', latency)
    write_csv(tables / 'latency_summary.csv', [means])
    for test_id, rows in candidate_tables.items():
        write_csv(tables / f'{test_id.lower()}_candidates.csv', rows)

    for answer in answers:
        test_id = re.match(r'### (Q\d+):', answer).group(1)
        context_rows = []
        for label in ('Baseline', 'Re-ranked'):
            match = re.search(r'\*\*' + label + r' Top 3:\*\* ([^\n]+)', answer)
            context_rows.append({'pipeline': label, 'context_chunk_ids': match.group(1)})
        write_csv(tables / f'{test_id.lower()}_context.csv', context_rows)

    (RESULTS / 'answer_comparisons.md').write_text('\n\n---\n\n'.join(answers) + '\n',
                                                   encoding='utf-8')
    (graphs / 'notebook_latency.png').write_bytes(base64.b64decode(original_plot))

    # Create publication-resolution graphs from the recorded display values.
    # No model inference or API call is involved.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    baseline = [float(r['baseline_retrieval_ms']) for r in latency]
    rerank = [float(r['cross_encoder_reranking_ms']) for r in latency]
    fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
    names = [r['test_id'] for r in latency]
    ax.bar(names, baseline, label='Baseline retrieval', color='#24466B', width=.62)
    ax.bar(names, rerank, bottom=baseline, label='Additional re-ranking',
           color='#DA9652', width=.62)
    ax.set(ylabel='Median component time (ms)', xlabel='Stress-test query', ylim=(0, 525))
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E7EB')
    ax.legend(frameon=False, ncols=2, loc='upper center')
    fig.savefig(graphs / 'latency_breakdown.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.3), layout='constrained')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E7EB')
    for row, color in zip(qualifying, ['#24466B', '#B56D24']):
        ax.plot([0, 1], [row['baseline_rank'], row['reranked_rank']], marker='o',
                markersize=7, linewidth=2.5, color=color,
                label=f"{row['query']}: {row['chunk_id']}")
        ax.annotate(str(row['baseline_rank']), (0, row['baseline_rank']),
                    xytext=(-10, 0), textcoords='offset points', va='center', ha='right')
    ax.axhspan(.5, 3.5, color='#DDEBDC', alpha=.6, label='Top-3 context range')
    ax.set(xticks=[0, 1], xticklabels=['Baseline retrieval', 'After re-ranking'],
           ylabel='Rank (lower is better)', xlim=(-.25, 1.25), ylim=(10.7, .3))
    ax.set_yticks(range(1, 11))
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    fig.savefig(graphs / 'rank_shifts.png', dpi=300)
    plt.close(fig)

    manifest = {
        'source_notebook': NOTEBOOK.name,
        'source_sha256': hashlib.sha256(notebook_bytes).hexdigest(),
        'provenance': 'Saved notebook HTML tables, Markdown answers, and embedded PNG; no rerun.',
        'precision': 'Tables and redrawn graphs use the precision serialized by Jupyter displays.',
        'latency_protocol': 'One warm-up, five repetitions; component medians; total=sum of medians.',
        'files': {},
    }
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file() and path.suffix in ('.csv', '.png', '.md'):
            manifest['files'][path.relative_to(RESULTS).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(ranks)} query results, {len(qualifying)} rank shifts, '
          f'{len(latency)} timing rows, and {len(answers)} answer pairs.')
    print('Results folder:', RESULTS)


if __name__ == '__main__':
    main()
