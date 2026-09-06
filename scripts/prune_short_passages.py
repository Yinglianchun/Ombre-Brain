"""Back up and prune short owners when raising the passage length threshold.

Run with passage writers stopped. Reads canonical owners; writes only the
derived passage DB. Unchanged long-owner vectors retain their exact bytes.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.passage_shadow import PassageShadowIndex


def prune(index, owners, *, previous_min_chars=200, apply=False, backup_path=None):
    previous = PassageShadowIndex({}, index.embedding_engine)
    previous.passage_config = replace(index.passage_config, min_owner_chars=previous_min_chars)
    if previous_min_chars > index.passage_config.min_owner_chars:
        raise ValueError('This operation only raises the owner length threshold')
    owner_map = {(kind, owner_id): (body, title) for kind, owner_id, body, title in owners}
    with closing(sqlite3.connect(Path(index.db_path).resolve().as_uri()+'?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute('SELECT * FROM memory_passage_owner_state'))
        counts = {(r[0], r[1]): r[2] for r in conn.execute(
            'SELECT owner_kind,owner_id,count(*) FROM memory_passage_embeddings GROUP BY owner_kind,owner_id')}
        states = {(r['owner_kind'], r['owner_id']): r for r in rows}
        changes, unresolved, unchanged = [], [], []
        for key in sorted(states.keys() | counts.keys()):
            if key not in owner_map:
                unresolved.append(':'.join(key))
                continue
            body, title = owner_map[key]
            state = states.get(key)
            source_hash = index._source_hash(*key, body, title)
            short = len(body.strip()) <= index.passage_config.min_owner_chars
            if short:
                changes.append((key, source_hash, True, counts.get(key, 0)))
            elif state and state['source_hash'] == previous._source_hash(*key, body, title):
                # Only the eligibility threshold changed; the text and split
                # layout inputs remain identical, so no model call is needed.
                changes.append((key, source_hash, False, 0))
            else:
                unchanged.append(':'.join(key))
        report = {'mode': 'apply' if apply else 'plan',
                  'min_owner_chars': index.passage_config.min_owner_chars,
                  'before_passages': sum(counts.values()),
                  'short_owners_with_passages': sum(short and count > 0 for _,_,short,count in changes),
                  'passages_to_remove': sum(count for _,_,_,count in changes),
                  'long_owner_signatures_updated': sum(not short for _,_,short,_ in changes),
                  'unresolved_owners': unresolved, 'unchanged_long_owners': unchanged,
                  'canonical_writes': 0, 'embedding_requests': 0}
        if not apply:
            return report
        if unresolved:
            raise ValueError('Resolve indexed owners before applying: '+', '.join(unresolved))
        backup = Path(backup_path or (index.db_path+'.before-min-'+str(index.passage_config.min_owner_chars)+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.bak'))
        if backup.exists():
            raise FileExistsError(backup)
        with closing(sqlite3.connect(backup)) as target:
            conn.backup(target)
        report['backup'] = str(backup)
    with closing(sqlite3.connect(Path(index.db_path).resolve().as_uri()+'?mode=rw', uri=True)) as conn:
        with conn:
            for (kind, owner_id), source_hash, short, _ in changes:
                if short:
                    conn.execute('DELETE FROM memory_passage_embeddings WHERE owner_kind=? AND owner_id=?', (kind,owner_id))
                    conn.execute('INSERT OR REPLACE INTO memory_passage_owner_state VALUES (?,?,?,?,?)',
                                 (kind,owner_id,source_hash,0,datetime.now(timezone.utc).isoformat()))
                else:
                    conn.execute('UPDATE memory_passage_embeddings SET source_hash=? WHERE owner_kind=? AND owner_id=?', (source_hash,kind,owner_id))
                    conn.execute('UPDATE memory_passage_owner_state SET source_hash=? WHERE owner_kind=? AND owner_id=?', (source_hash,kind,owner_id))
        report['after_passages'] = conn.execute('SELECT count(*) FROM memory_passage_embeddings').fetchone()[0]
    return report


async def main(args):
    from bucket_manager import BucketManager
    from utils import load_config, bucket_text_for_embedding
    config = load_config(args.config or None)
    embedding = config.get('embedding', {})
    engine = SimpleNamespace(model=embedding.get('model', ''), document_instruction=embedding.get('document_instruction', ''))
    index = PassageShadowIndex(config, engine)
    buckets = await BucketManager(config).list_all(include_archive=True)
    scenes = [{'id': b['id'], 'content': bucket_text_for_embedding(b)} for b in buckets]
    event_db = Path(index.db_path).parent/'fact_events.sqlite'
    with closing(sqlite3.connect(event_db.as_uri()+'?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        events = [dict(row) for row in conn.execute("SELECT item_id,title,body FROM fact_events WHERE item_type='event'")]
    report = prune(index, index._owners(scenes, events), previous_min_chars=args.previous_min_chars, apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='')
    parser.add_argument('--previous-min-chars', type=int, default=200)
    parser.add_argument('--apply', action='store_true')
    asyncio.run(main(parser.parse_args()))
