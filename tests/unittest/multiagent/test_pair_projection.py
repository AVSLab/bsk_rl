"""The login reader must skip large learner arrays without materializing them."""

import io
import json
import tracemalloc

import pytest

from examples.multiagent_imaging.cluster.collect_pair_comparison import project, skip_value
from examples.multiagent_imaging.update_evidence import JSONStream


@pytest.mark.parametrize('size', [65530, 65531, 65532, 65533, 65534, 65535, 65536])
def test_skip_escaped_delimiters_across_chunk_boundaries(size):
    value = {'drop': ['x' * size + '\\"}[]', {'inside': ['\\\\', '\"']}], 'keep': 19}
    reader = JSONStream(io.StringIO(json.dumps(value)))
    assert project(reader, {'keep'}) == {'keep': 19}
    reader.finish()


def test_skip_large_array_keeps_reader_memory_bounded():
    raw = '{"drop":[' + ','.join(['[1,2,3,4,5,6,7,8,9]'] * 200000) + '],"keep":true}'
    stream = io.StringIO(raw)
    tracemalloc.start()
    try:
        reader = JSONStream(stream)
        assert project(reader, {'keep'}) == {'keep': True}
        reader.finish()
        assert tracemalloc.get_traced_memory()[1] < 1024 * 1024
    finally:
        tracemalloc.stop()


def test_truncated_skipped_container_is_rejected():
    reader = JSONStream(io.StringIO('{"drop":[{"x":"escaped\\"end"}'))
    with pytest.raises(ValueError, match='Truncated'):
        project(reader, {'keep'})


def test_selected_values_and_array_projection_survive_skipping():
    raw = json.dumps({'ignore': [{'a': [1, 2]}], 'capture_products': [{'id': 3, 'extra': 9}], 'a': 5})
    reader = JSONStream(io.StringIO(raw))
    assert project(reader, {'a'}, {'capture_products': lambda x: x['id']}) == {'a': 5, 'capture_products': [3]}
    reader.finish()
