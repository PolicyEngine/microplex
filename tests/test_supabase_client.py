"""
TDD tests for batch Supabase operations.

These tests verify that batch operations:
1. Correctly insert/upsert multiple records at once
2. Handle conflicts (duplicates) gracefully
3. Are significantly faster than individual operations
"""

import pytest
import time
import os
from unittest.mock import Mock, patch, MagicMock
import responses
import json

# Import the Supabase client
try:
    from scripts.load_pe_targets import BatchSupabaseClient as SupabaseClient
except ImportError:
    SupabaseClient = None


SUPABASE_URL = "https://test.supabase.co"
SUPABASE_KEY = "test-key"


class TestBatchUpsertStrata:
    """Tests for batch stratum upsert operations."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        if SupabaseClient is None:
            pytest.skip("SupabaseClient not yet implemented")
        return SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

    @responses.activate
    def test_batch_upsert_strata_inserts_multiple(self, client):
        """batch_upsert_strata should insert multiple strata in one request."""
        strata = [
            {"name": "Test stratum 1", "jurisdiction": "us", "description": "Desc 1"},
            {"name": "Test stratum 2", "jurisdiction": "us", "description": "Desc 2"},
            {"name": "Test stratum 3", "jurisdiction": "us-ca", "description": "Desc 3"},
        ]

        # Mock the upsert endpoint
        responses.add(
            responses.POST,
            f"{SUPABASE_URL}/rest/v1/strata",
            json=[
                {"id": "uuid-1", **strata[0]},
                {"id": "uuid-2", **strata[1]},
                {"id": "uuid-3", **strata[2]},
            ],
            status=201,
        )

        result = client.batch_upsert_strata(strata)

        # Should return list of inserted records with IDs
        assert len(result) == 3
        assert result[0]["id"] == "uuid-1"
        assert result[1]["id"] == "uuid-2"
        assert result[2]["id"] == "uuid-3"

        # Should have made only ONE request
        assert len(responses.calls) == 1

        # Request should contain all strata
        request_body = json.loads(responses.calls[0].request.body)
        assert len(request_body) == 3

    @responses.activate
    def test_batch_upsert_strata_handles_conflicts(self, client):
        """batch_upsert_strata should handle duplicates via upsert."""
        strata = [
            {"name": "Existing stratum", "jurisdiction": "us", "description": "Updated"},
        ]

        # Mock upsert with conflict resolution
        responses.add(
            responses.POST,
            f"{SUPABASE_URL}/rest/v1/strata",
            json=[{"id": "existing-uuid", **strata[0]}],
            status=200,  # 200 for upsert (vs 201 for insert)
        )

        result = client.batch_upsert_strata(strata)

        assert len(result) == 1
        assert result[0]["id"] == "existing-uuid"

        # Check upsert headers were set
        request_headers = responses.calls[0].request.headers
        assert "resolution=merge-duplicates" in request_headers.get("Prefer", "")

    @responses.activate
    def test_batch_upsert_strata_returns_ids_mapping(self, client):
        """batch_upsert_strata should return a name->id mapping for easy lookup."""
        strata = [
            {"name": "Stratum A", "jurisdiction": "us", "description": "A"},
            {"name": "Stratum B", "jurisdiction": "us", "description": "B"},
        ]

        responses.add(
            responses.POST,
            f"{SUPABASE_URL}/rest/v1/strata",
            json=[
                {"id": "id-a", **strata[0]},
                {"id": "id-b", **strata[1]},
            ],
            status=201,
        )

        result = client.batch_upsert_strata(strata, return_mapping=True)

        # Should return dict mapping (name, jurisdiction) -> id
        assert result[("Stratum A", "us")] == "id-a"
        assert result[("Stratum B", "us")] == "id-b"


class TestBatchUpsertTargets:
    """Tests for batch target upsert operations."""

    @pytest.fixture
    def client(self):
        if SupabaseClient is None:
            pytest.skip("SupabaseClient not yet implemented")
        return SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

    @responses.activate
    def test_batch_upsert_targets_inserts_multiple(self, client):
        """batch_upsert_targets should insert multiple targets in one request."""
        targets = [
            {"source_id": "src-1", "stratum_id": "str-1", "variable": "income",
             "value": 1000000, "target_type": "amount", "period": 2024},
            {"source_id": "src-1", "stratum_id": "str-2", "variable": "population",
             "value": 50000, "target_type": "count", "period": 2024},
        ]

        responses.add(
            responses.POST,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[{"id": f"target-{i}", **t} for i, t in enumerate(targets)],
            status=201,
        )

        result = client.batch_upsert_targets(targets)

        assert len(result) == 2
        assert len(responses.calls) == 1

    @responses.activate
    def test_batch_upsert_targets_chunks_large_batches(self, client):
        """batch_upsert_targets should chunk very large batches to avoid timeouts."""
        # Create 1500 targets (should be split into chunks)
        targets = [
            {"source_id": "src-1", "stratum_id": f"str-{i}", "variable": "var",
             "value": i * 100, "target_type": "amount", "period": 2024}
            for i in range(1500)
        ]

        # Mock multiple chunk requests
        for _ in range(3):  # Expect 3 chunks of 500
            responses.add(
                responses.POST,
                f"{SUPABASE_URL}/rest/v1/targets",
                json=[{"id": f"id-{i}"} for i in range(500)],
                status=201,
            )

        result = client.batch_upsert_targets(targets, chunk_size=500)

        # Should have made 3 requests (1500 / 500)
        assert len(responses.calls) == 3

    @responses.activate
    def test_batch_upsert_targets_uses_composite_key(self, client):
        """batch_upsert_targets should upsert on (source_id, stratum_id, variable, period)."""
        targets = [
            {"source_id": "src-1", "stratum_id": "str-1", "variable": "income",
             "value": 2000000, "target_type": "amount", "period": 2024},
        ]

        responses.add(
            responses.POST,
            f"{SUPABASE_URL}/rest/v1/targets",
            json=[{"id": "existing-id", **targets[0]}],
            status=200,
        )

        client.batch_upsert_targets(targets)

        # Check on_conflict parameter in URL or header
        request = responses.calls[0].request
        assert "on_conflict" in request.url or "resolution=merge-duplicates" in request.headers.get("Prefer", "")


class TestBatchPerformance:
    """Tests verifying batch operations are faster than individual ones."""

    @pytest.fixture
    def client(self):
        if SupabaseClient is None:
            pytest.skip("SupabaseClient not yet implemented")
        return SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

    @responses.activate
    def test_batch_is_faster_than_individual(self, client):
        """Batch insert of N records should make fewer requests than N individual inserts."""
        n_records = 100
        strata = [
            {"name": f"Stratum {i}", "jurisdiction": "us", "description": f"Desc {i}"}
            for i in range(n_records)
        ]

        # Mock batch endpoint
        responses.add(
            responses.POST,
            f"{SUPABASE_URL}/rest/v1/strata",
            json=[{"id": f"id-{i}", **s} for i, s in enumerate(strata)],
            status=201,
        )

        client.batch_upsert_strata(strata)

        # Batch should make 1 request, not 100
        assert len(responses.calls) == 1


class TestBatchIntegration:
    """Integration tests using real Supabase (skipped if no credentials)."""

    @pytest.fixture
    def live_client(self):
        """Create a client connected to real Supabase."""
        url = os.environ.get("POLICYENGINE_SUPABASE_URL") or os.environ.get(
            "SUPABASE_URL"
        )
        key = os.environ.get("POLICYENGINE_SUPABASE_SERVICE_KEY")
        if not url or not key:
            pytest.skip("No Supabase credentials - skipping integration test")
        if SupabaseClient is None:
            pytest.skip("SupabaseClient not yet implemented")
        return SupabaseClient(url, key)

    def test_batch_upsert_strata_live(self, live_client):
        """Test batch upsert against real Supabase."""
        import uuid
        test_id = str(uuid.uuid4())[:8]

        strata = [
            {"name": f"Test batch {test_id} - 1", "jurisdiction": "test",
             "description": "Integration test stratum 1"},
            {"name": f"Test batch {test_id} - 2", "jurisdiction": "test",
             "description": "Integration test stratum 2"},
        ]

        result = live_client.batch_upsert_strata(strata)

        assert len(result) == 2
        assert all("id" in r for r in result)

        # Cleanup - delete test strata
        for r in result:
            live_client._request_with_retry(
                "DELETE",
                f"{live_client.base_url}/strata?id=eq.{r['id']}",
                headers=live_client.headers
            )
