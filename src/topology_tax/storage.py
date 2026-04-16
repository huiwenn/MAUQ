import json
import sqlite3
from pathlib import Path
from typing import Optional

class ResultStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                dataset TEXT NOT NULL, question_id TEXT NOT NULL, topology TEXT NOT NULL,
                run_id INTEGER NOT NULL, agent_id TEXT NOT NULL, agent_position INTEGER NOT NULL,
                prompt TEXT, response TEXT, answer_extracted TEXT, correct BOOLEAN,
                model_id TEXT, temperature REAL, latency_ms INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (dataset, question_id, topology, run_id, agent_id)
            )
        """)
        self.conn.commit()

    def save_result(self, **kwargs):
        metadata = kwargs.pop("metadata", {})
        kwargs["metadata"] = json.dumps(metadata) if metadata else "{}"
        self.conn.execute("""
            INSERT OR REPLACE INTO results
            (dataset, question_id, topology, run_id, agent_id, agent_position,
             prompt, response, answer_extracted, correct, model_id, temperature,
             latency_ms, input_tokens, output_tokens, metadata)
            VALUES (:dataset, :question_id, :topology, :run_id, :agent_id,
                    :agent_position, :prompt, :response, :answer_extracted,
                    :correct, :model_id, :temperature, :latency_ms,
                    :input_tokens, :output_tokens, :metadata)
        """, kwargs)
        self.conn.commit()

    def is_completed(self, dataset, question_id, topology, run_id, agent_id):
        row = self.conn.execute(
            "SELECT 1 FROM results WHERE dataset=? AND question_id=? AND topology=? AND run_id=? AND agent_id=?",
            (dataset, question_id, topology, run_id, agent_id)).fetchone()
        return row is not None

    def get_results(self, dataset=None, question_id=None, topology=None):
        conditions, params = [], []
        if dataset: conditions.append("dataset=?"); params.append(dataset)
        if question_id: conditions.append("question_id=?"); params.append(question_id)
        if topology: conditions.append("topology=?"); params.append(topology)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.conn.execute(f"SELECT * FROM results WHERE {where}", params).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
