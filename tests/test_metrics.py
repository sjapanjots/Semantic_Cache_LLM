import pytest

from semcache import SemanticCache
from semcache.metrics import collect, prometheus_text


@pytest.fixture
def cache():
    c = SemanticCache(pricing={"model-a": (1.0, 2.0)})
    c.set(
        "How do I reverse a list in Python?",
        "x",
        model="model-a",
        input_tokens=1000,
        output_tokens=500,
    )  # costs $0.002
    c.get("python: how to reverse a list", model="model-a")  # semantic hit
    c.get("How do I reverse a list in Python?", model="model-a")  # exact hit
    c.get("What is the best pizza topping?", model="model-a")  # miss
    return c


def by_name(metrics):
    return {m.name: m for m in metrics}


def test_collect_values(cache):
    m = by_name(collect(cache))
    assert m["lookups"].samples == (({}, 3.0),)
    assert dict((s[0]["kind"], s[1]) for s in m["hits"].samples) == {"exact": 1, "semantic": 1}
    assert m["misses"].samples[0][1] == 1
    assert m["writes"].samples[0][1] == 1
    assert dict((s[0]["type"], s[1]) for s in m["tokens_saved"].samples) == {
        "input": 2000,
        "output": 1000,
    }
    assert m["cost_saved_usd"].samples[0][1] == pytest.approx(0.004)
    assert m["cost_spent_usd"].samples[0][1] == pytest.approx(0.002)
    assert m["hit_ratio"].samples[0][1] == pytest.approx(2 / 3)
    assert m["entries"].samples[0][1] == 1
    assert m["model_hits"].samples == (({"model": "model-a"}, 2.0),)
    assert m["lookups"].kind == "counter" and m["hit_ratio"].kind == "gauge"


def test_collect_without_entries_or_models():
    names = set(by_name(collect(SemanticCache(), include_entries=False)))
    assert "entries" not in names and "model_hits" not in names
    assert {"lookups", "hits", "cost_saved_usd"} <= names


def test_prometheus_text_format(cache):
    text = prometheus_text(cache)
    assert text.endswith("\n")
    assert "# TYPE semcache_lookups_total counter\nsemcache_lookups_total 3\n" in text
    assert 'semcache_hits_total{kind="semantic"} 1\n' in text
    assert "# TYPE semcache_hit_ratio gauge" in text
    assert "semcache_cost_saved_usd_total 0.004\n" in text
    assert 'semcache_model_hits_total{model="model-a"} 2\n' in text
    assert "semcache_entries 1\n" in text
    assert "myapp_lookups_total 3" in prometheus_text(cache, prefix="myapp")


def test_prometheus_text_escapes_labels():
    c = SemanticCache()
    c.get('weird "model"\\name', model='weird "model"\\name')
    assert 'model="weird \\"model\\"\\\\name"' in prometheus_text(c)


def test_prometheus_text_parses_with_official_parser(cache):
    parser = pytest.importorskip("prometheus_client.parser")
    families = {f.name: f for f in parser.text_string_to_metric_families(prometheus_text(cache))}
    assert families["semcache_lookups"].type == "counter"
    assert families["semcache_lookups"].samples[0].value == 3
    assert families["semcache_hit_ratio"].type == "gauge"


def test_prometheus_collector(cache):
    prometheus_client = pytest.importorskip("prometheus_client")
    from semcache.metrics import PrometheusCollector

    registry = prometheus_client.CollectorRegistry()
    registry.register(PrometheusCollector(cache))
    assert registry.get_sample_value("semcache_lookups_total") == 3
    assert registry.get_sample_value("semcache_hits_total", {"kind": "exact"}) == 1
    assert registry.get_sample_value("semcache_cost_saved_usd_total") == pytest.approx(0.004)
    assert registry.get_sample_value("semcache_model_hits_total", {"model": "model-a"}) == 2
    assert registry.get_sample_value("semcache_entries") == 1
    cache.get("How do I reverse a list in Python?", model="model-a")
    assert registry.get_sample_value("semcache_lookups_total") == 4  # read live on scrape


def test_prometheus_collector_name_clash_detected(cache):
    prometheus_client = pytest.importorskip("prometheus_client")
    from semcache.metrics import PrometheusCollector

    registry = prometheus_client.CollectorRegistry()
    registry.register(PrometheusCollector(cache))
    with pytest.raises(ValueError):
        registry.register(PrometheusCollector(cache))


def test_opentelemetry(cache):
    pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from semcache.metrics import register_opentelemetry

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    instruments = register_opentelemetry(cache, provider.get_meter("test"))
    assert len(instruments) == 14

    def points():
        metrics = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics
        return {
            m.name: {tuple(sorted(p.attributes.items())): p.value for p in m.data.data_points}
            for m in metrics
        }

    data = points()
    assert data["semcache.lookups"] == {(): 3}
    assert data["semcache.hits"] == {(("kind", "exact"),): 1, (("kind", "semantic"),): 1}
    assert data["semcache.cost_saved_usd"][()] == pytest.approx(0.004)
    assert data["semcache.model_hits"] == {(("model", "model-a"),): 2}
    assert data["semcache.entries"] == {(): 1}
    cache.get("What is the best pizza topping?", model="model-a")
    assert points()["semcache.misses"] == {(): 2}
