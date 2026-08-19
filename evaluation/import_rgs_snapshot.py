"""Decode an LDK Rapid Gossip Sync snapshot into reproducible experiment data.

The decoder follows the public RGS v1/v2 wire format implemented by
``lightning-rapid-gossip-sync``.  It preserves every channel and directional
update in a CSV, then produces the simple, capacity-weighted edge list expected
by the Ballast simulator. Parallel channels are summed and only the largest
connected component is exported because the simulator samples arbitrary node
pairs and uses an undirected simple graph.
"""

import argparse
import csv
import datetime as dt
import json
import os
import struct
from collections import defaultdict


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        end = self.pos + count
        if end > len(self.data):
            raise ValueError(f"short read at byte {self.pos}: need {count}")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def uint(self, count):
        return int.from_bytes(self.take(count), "big")

    def u8(self):
        return self.uint(1)

    def u16(self):
        return self.uint(2)

    def u32(self):
        return self.uint(4)

    def u64(self):
        return self.uint(8)

    def bigsize(self):
        prefix = self.u8()
        if prefix < 0xFD:
            return prefix
        if prefix == 0xFD:
            value = self.u16()
            if value < 0xFD:
                raise ValueError("non-canonical BigSize u16")
            return value
        if prefix == 0xFE:
            value = self.u32()
            if value < 0x10000:
                raise ValueError("non-canonical BigSize u32")
            return value
        value = self.u64()
        if value < 0x100000000:
            raise ValueError("non-canonical BigSize u64")
        return value

    def feature_bits(self):
        # LDK Features::write uses a big-endian u16 byte length.
        return self.take(self.u16())

    def byte_vector(self):
        # LDK Writeable for Vec<u8> uses a big-endian u16 byte length.
        return self.take(self.u16())


def read_node_ids(reader, version):
    defaults = []
    if version >= 2:
        defaults = [reader.feature_bits() for _ in range(reader.u8())]

    count = reader.u32()
    node_ids = []
    detail_counts = defaultdict(int)
    for _ in range(count):
        encoded = bytearray(reader.take(33))
        if version < 2:
            node_ids.append(bytes(encoded).hex())
            continue

        flags = encoded[0]
        has_addresses = bool(flags & 0x04)
        feature_marker = (flags >> 3) & 0x07
        is_reminder = bool(flags & 0x40)
        has_extra = bool(flags & 0x80)
        encoded[0] &= 0x03
        node_ids.append(bytes(encoded).hex())

        if has_addresses:
            detail_counts["address_updates"] += 1
            for _ in range(reader.u8()):
                reader.take(reader.u8())
        if feature_marker:
            detail_counts["feature_updates"] += 1
            if feature_marker == 7:
                reader.feature_bits()
            elif feature_marker > len(defaults):
                raise ValueError(f"invalid node feature default {feature_marker}")
        if is_reminder:
            detail_counts["reminders"] += 1
        if has_extra:
            detail_counts["additional_data"] += 1
            reader.byte_vector()
    return node_ids, dict(detail_counts)


def read_announcements(reader, version, node_ids):
    count = reader.u32()
    channels = []
    previous_scid = 0
    for _ in range(count):
        features = reader.feature_bits()
        scid = previous_scid + reader.bigsize()
        previous_scid = scid
        node1 = reader.bigsize()
        node2_encoded = reader.bigsize()
        has_extra = bool(node2_encoded & (1 << 63))
        node2 = node2_encoded & ~(1 << 63)
        if node1 >= len(node_ids) or node2 >= len(node_ids):
            raise ValueError(f"channel {scid} has out-of-range node index")

        funding_sats = None
        if version >= 2 and has_extra:
            extra = Reader(reader.byte_vector())
            funding_sats = extra.bigsize()
        channels.append({
            "scid": scid,
            "node1_index": node1,
            "node2_index": node2,
            "node1_pubkey": node_ids[node1],
            "node2_pubkey": node_ids[node2],
            "funding_sats": funding_sats,
            "feature_bytes": len(features),
            "dir0": None,
            "dir1": None,
        })
    return channels


def read_updates(reader, version):
    count = reader.u32()
    if count == 0:
        return {}, 0

    defaults = {
        "cltv_delta": reader.u16(),
        "htlc_min_msat": reader.u64(),
        "fee_base_msat": reader.u32(),
        "fee_ppm": reader.u32(),
        "htlc_max_msat": reader.u64(),
    }
    updates = {}
    previous_scid = 0
    previous_direction = None
    extra_records = 0
    for _ in range(count):
        delta = reader.bigsize()
        scid = previous_scid + delta
        previous_scid = scid
        flags = reader.u8()
        direction = flags & 1

        if version >= 2 and delta == 0 and direction == previous_direction:
            reader.byte_vector()
            extra_records += 1
            continue
        previous_direction = direction

        key = (scid, direction)
        incremental = bool(flags & 0x80)
        values = dict(updates.get(key, defaults) if incremental else defaults)
        values.update({
            "disabled": bool(flags & 0x02),
            "incremental": incremental,
        })
        if flags & 0x40:
            values["cltv_delta"] = reader.u16()
        if flags & 0x20:
            values["htlc_min_msat"] = reader.u64()
        if flags & 0x10:
            values["fee_base_msat"] = reader.u32()
        if flags & 0x08:
            values["fee_ppm"] = reader.u32()
        if flags & 0x04:
            values["htlc_max_msat"] = reader.u64()
        updates[key] = values
    return updates, extra_records


def connected_components(edges):
    parent = {}
    size = {}

    def find(node):
        parent.setdefault(node, node)
        size.setdefault(node, 1)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in edges:
        union(a, b)
    groups = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)
    return sorted(groups.values(), key=lambda group: (-len(group), min(group)))


def attach_updates(channels, updates):
    by_scid = {channel["scid"]: channel for channel in channels}
    unknown = 0
    for (scid, direction), values in updates.items():
        channel = by_scid.get(scid)
        if channel is None:
            unknown += 1
        else:
            channel[f"dir{direction}"] = values
    return unknown


def write_channels(path, channels):
    fields = [
        "scid", "node1_index", "node2_index", "node1_pubkey", "node2_pubkey",
        "funding_sats", "feature_bytes",
    ]
    directional = ["disabled", "incremental", "cltv_delta", "htlc_min_msat",
                   "fee_base_msat", "fee_ppm", "htlc_max_msat"]
    fields += [f"dir{direction}_{field}" for direction in (0, 1)
               for field in directional]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for channel in channels:
            row = {field: channel.get(field, "") for field in fields}
            for direction in (0, 1):
                values = channel[f"dir{direction}"] or {}
                for field in directional:
                    row[f"dir{direction}_{field}"] = values.get(field, "")
            writer.writerow(row)


def build_edge_list(channels):
    aggregated = defaultdict(int)
    included_channels = 0
    excluded_disabled = 0
    excluded_no_capacity = 0
    for channel in channels:
        capacity = channel["funding_sats"]
        if capacity is None or capacity <= 0:
            excluded_no_capacity += 1
            continue
        directions = [channel["dir0"], channel["dir1"]]
        if all(direction is not None and direction["disabled"] for direction in directions):
            excluded_disabled += 1
            continue
        a, b = sorted((channel["node1_index"], channel["node2_index"]))
        if a != b:
            aggregated[(a, b)] += capacity
            included_channels += 1

    components = connected_components(aggregated)
    largest = components[0] if components else set()
    lcc_edges = {edge: capacity for edge, capacity in aggregated.items()
                 if edge[0] in largest and edge[1] in largest}
    return lcc_edges, {
        "unique_node_pairs_before_lcc": len(aggregated),
        "connected_components": len(components),
        "largest_component_nodes": len(largest),
        "largest_component_edges": len(lcc_edges),
        "excluded_both_directions_disabled": excluded_disabled,
        "excluded_missing_or_zero_capacity": excluded_no_capacity,
        "included_channel_announcements": included_channels,
        "parallel_channels_collapsed": included_channels - len(aggregated),
    }


def main():
    parser = argparse.ArgumentParser(description="decode an LDK RGS v1/v2 snapshot")
    parser.add_argument("--input", required=True)
    parser.add_argument("--edge-output", required=True)
    parser.add_argument("--channel-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        data = handle.read()
    reader = Reader(data)
    if reader.take(3) != b"LDK":
        raise SystemExit("not an LDK RGS snapshot")
    version = reader.u8()
    if version not in (1, 2):
        raise SystemExit(f"unsupported RGS version {version}")
    chain_hash = reader.take(32).hex()
    timestamp = reader.u32()
    node_ids, node_details = read_node_ids(reader, version)
    channels = read_announcements(reader, version, node_ids)
    updates, extra_update_records = read_updates(reader, version)
    unknown_updates = attach_updates(channels, updates)
    if reader.pos != len(data):
        raise SystemExit(f"decoder left {len(data) - reader.pos} bytes unread")

    write_channels(args.channel_output, channels)
    edges, graph_stats = build_edge_list(channels)
    os.makedirs(os.path.dirname(os.path.abspath(args.edge_output)), exist_ok=True)
    with open(args.edge_output, "w") as handle:
        for (a, b), capacity in sorted(edges.items()):
            handle.write(f"{a} {b} {capacity}\n")

    metadata = {
        "source_file": os.path.basename(args.input),
        "source_bytes": len(data),
        "rgs_version": version,
        "chain_hash": chain_hash,
        "latest_seen_timestamp": timestamp,
        "latest_seen_utc": dt.datetime.fromtimestamp(
            timestamp, tz=dt.timezone.utc).isoformat(),
        "node_id_count": len(node_ids),
        "channel_announcement_count": len(channels),
        "channel_update_count": len(updates),
        "channel_update_extra_records": extra_update_records,
        "updates_for_unknown_channels": unknown_updates,
        "channels_with_funding_capacity": sum(
            channel["funding_sats"] is not None for channel in channels),
        "channels_with_two_directional_updates": sum(
            channel["dir0"] is not None and channel["dir1"] is not None
            for channel in channels),
        "node_details": node_details,
        "graph_conversion": graph_stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.metadata_output)), exist_ok=True)
    with open(args.metadata_output, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
