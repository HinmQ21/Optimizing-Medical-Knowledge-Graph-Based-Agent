"""Step 4: Data model (dataclasses) + build hypergraph + save JSON."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .verbalize import MedicalTemplateEngine


@dataclass
class Entity:
    name: str
    entity_type: str  # disease, drug, gene/protein, pathway, ...


@dataclass
class Hyperedge:
    id: str  # "he_{i:06d}"
    description: str  # verbalized text
    entities: list[str]  # entity names
    hedge_type: str  # neighbor_agg | composite | path
    source_relation: str  # relation or path_pattern
    anchor: str = ""  # anchor entity (if applicable)


@dataclass
class MedicalHypergraph:
    entities: dict[str, Entity] = field(default_factory=dict)  # name -> Entity
    hyperedges: list[Hyperedge] = field(default_factory=list)
    entity_to_hedges: dict[str, list[str]] = field(
        default_factory=dict
    )  # entity_name -> [hedge_id, ...]


def build_hypergraph(
    all_hyperedges_raw: list[dict], engine: MedicalTemplateEngine
) -> MedicalHypergraph:
    """Build MedicalHypergraph from raw aggregated hyperedge dicts."""
    entities = {}
    hyperedges = []
    entity_to_hedges: dict[str, list[str]] = {}

    for i, he_raw in enumerate(all_hyperedges_raw):
        desc = engine.verbalize(he_raw)

        # Collect entities based on hyperedge type
        if he_raw['type'] == 'neighbor_agg':
            ent_list = [he_raw['anchor']] + he_raw['neighbors']
            types = [he_raw['anchor_type']] + he_raw['neighbor_types']
        elif he_raw['type'] == 'composite':
            ent_list = he_raw['entities']
            types = [he_raw['anchor_type']] + ['unknown'] * (len(ent_list) - 1)
        else:  # path
            ent_list = he_raw['entities']
            types = he_raw['entity_types']

        for name, etype in zip(ent_list, types):
            if name not in entities:
                entities[name] = Entity(name=name, entity_type=etype)

        hedge_id = f"he_{i:06d}"
        he = Hyperedge(
            id=hedge_id,
            description=desc,
            entities=ent_list,
            hedge_type=he_raw['type'],
            source_relation=he_raw.get('relation') or he_raw.get('path_pattern', ''),
            anchor=he_raw.get('anchor', ''),
        )
        hyperedges.append(he)

        for name in ent_list:
            entity_to_hedges.setdefault(name, []).append(hedge_id)

    return MedicalHypergraph(
        entities=entities,
        hyperedges=hyperedges,
        entity_to_hedges=entity_to_hedges,
    )


def save_hypergraph(hg: MedicalHypergraph, path: str = "data/medical_hg.json"):
    """Serialize MedicalHypergraph to JSON."""
    data = {
        "entities": {
            k: {"name": v.name, "type": v.entity_type} for k, v in hg.entities.items()
        },
        "hyperedges": [
            {
                "id": h.id,
                "description": h.description,
                "entities": h.entities,
                "type": h.hedge_type,
                "relation": h.source_relation,
                "anchor": h.anchor,
            }
            for h in hg.hyperedges
        ],
        "entity_to_hedges": hg.entity_to_hedges,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Saved hypergraph to {path}")
    print(f"  Entities: {len(hg.entities):,}")
    print(f"  Hyperedges: {len(hg.hyperedges):,}")
