"""
Tests for models/base.py - Base interfaces for synthesis models.

Following TDD: these tests define the expected behavior of the data structures
and model interfaces.
"""


import numpy as np
import pandas as pd
import pytest

from microplex.models.base import (
    BaseGraphModel,
    BaseSynthesisModel,
    BaseTrajectoryModel,
    ImputationResult,
    SyntheticPopulation,
)


class TestSyntheticPopulation:
    """Tests for SyntheticPopulation dataclass."""

    def test_minimal_population(self):
        """Population can be created with just persons DataFrame."""
        persons = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
        })

        pop = SyntheticPopulation(persons=persons)

        assert pop.persons is not None
        assert pop.households is None
        assert pop.edges is None
        assert pop.weights is None

    def test_n_persons_with_person_id(self):
        """n_persons should count unique person_ids."""
        persons = pd.DataFrame({
            "person_id": [1, 1, 2, 2, 3, 3],  # 3 persons, 2 periods each
            "period": [0, 1, 0, 1, 0, 1],
            "age": [25, 26, 30, 31, 35, 36],
        })

        pop = SyntheticPopulation(persons=persons)

        assert pop.n_persons == 3

    def test_n_persons_without_person_id(self):
        """Without person_id column, n_persons should be row count."""
        persons = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
        })

        pop = SyntheticPopulation(persons=persons)

        assert pop.n_persons == 4

    def test_n_households_with_households(self):
        """n_households should count unique household_ids."""
        persons = pd.DataFrame({"person_id": [1, 2, 3], "age": [25, 30, 35]})
        households = pd.DataFrame({
            "household_id": [1, 1, 2, 2],  # 2 households, 2 periods each
            "period": [0, 1, 0, 1],
            "total_income": [100000, 105000, 80000, 85000],
        })

        pop = SyntheticPopulation(persons=persons, households=households)

        assert pop.n_households == 2

    def test_n_households_without_households(self):
        """Without households DataFrame, n_households should be 0."""
        persons = pd.DataFrame({"person_id": [1, 2], "age": [25, 30]})

        pop = SyntheticPopulation(persons=persons)

        assert pop.n_households == 0

    def test_n_periods_with_period_column(self):
        """n_periods should count unique periods."""
        persons = pd.DataFrame({
            "person_id": [1, 1, 1, 2, 2, 2],
            "period": [0, 1, 2, 0, 1, 2],
            "age": [25, 26, 27, 30, 31, 32],
        })

        pop = SyntheticPopulation(persons=persons)

        assert pop.n_periods == 3

    def test_n_periods_without_period_column(self):
        """Without period column, n_periods should be 1."""
        persons = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [25, 30, 35],
        })

        pop = SyntheticPopulation(persons=persons)

        assert pop.n_periods == 1

    def test_to_cross_section_with_periods(self):
        """to_cross_section should extract specified period."""
        persons = pd.DataFrame({
            "person_id": [1, 1, 2, 2],
            "period": [0, 1, 0, 1],
            "age": [25, 26, 30, 31],
            "income": [50000, 52000, 60000, 62000],
        })

        pop = SyntheticPopulation(persons=persons)
        cs_0 = pop.to_cross_section(period=0)
        cs_1 = pop.to_cross_section(period=1)

        assert len(cs_0) == 2
        assert len(cs_1) == 2
        assert cs_0["age"].tolist() == [25, 30]
        assert cs_1["age"].tolist() == [26, 31]

    def test_to_cross_section_without_periods(self):
        """Without period column, to_cross_section returns full data."""
        persons = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [25, 30, 35],
        })

        pop = SyntheticPopulation(persons=persons)
        cs = pop.to_cross_section()

        assert len(cs) == 3
        pd.testing.assert_frame_equal(cs, persons)

    def test_to_cross_section_returns_copy(self):
        """to_cross_section should return a copy, not a view."""
        persons = pd.DataFrame({
            "person_id": [1, 2],
            "period": [0, 0],
            "age": [25, 30],
        })

        pop = SyntheticPopulation(persons=persons)
        cs = pop.to_cross_section()

        # Modify the cross-section
        cs.loc[cs.index[0], "age"] = 999

        # Original should be unchanged
        assert pop.persons.loc[pop.persons.index[0], "age"] == 25

    def test_with_weights(self):
        """Population can have calibration weights."""
        persons = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [25, 30, 35],
        })
        weights = np.array([1.0, 1.5, 0.8])

        pop = SyntheticPopulation(persons=persons, weights=weights)

        np.testing.assert_array_equal(pop.weights, weights)

    def test_with_edges(self):
        """Population can have household-person edges."""
        persons = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [25, 30, 5],
        })
        households = pd.DataFrame({
            "household_id": [1, 2],
            "total_income": [100000, 50000],
        })
        edges = pd.DataFrame({
            "household_id": [1, 1, 2],
            "person_id": [1, 3, 2],
            "relationship": ["head", "child", "head"],
        })

        pop = SyntheticPopulation(persons=persons, households=households, edges=edges)

        assert len(pop.edges) == 3


class TestImputationResult:
    """Tests for ImputationResult dataclass."""

    def test_basic_imputation_result(self):
        """ImputationResult should store samples and metadata."""
        samples = pd.DataFrame({
            "_input_row_id": [0, 0, 0, 1, 1, 1],
            "income": [50000, 51000, 49000, 60000, 62000, 58000],
            "wealth": [10000, 11000, 9000, 20000, 22000, 18000],
        })
        input_mask = pd.DataFrame({
            "income": [False, False],  # Not observed
            "wealth": [False, False],  # Not observed
        })

        result = ImputationResult(samples=samples, input_mask=input_mask, n_samples=3)

        assert result.n_samples == 3
        assert len(result.samples) == 6  # 2 rows * 3 samples

    def test_mean(self):
        """mean() should compute mean across samples for each input row."""
        samples = pd.DataFrame({
            "_input_row_id": [0, 0, 0, 1, 1, 1],
            "income": [48000, 50000, 52000, 58000, 60000, 62000],
        })
        input_mask = pd.DataFrame({"income": [False, False]})

        result = ImputationResult(samples=samples, input_mask=input_mask, n_samples=3)
        means = result.mean()

        assert len(means) == 2
        assert means.loc[0, "income"] == 50000  # (48k + 50k + 52k) / 3
        assert means.loc[1, "income"] == 60000

    def test_std(self):
        """std() should compute std across samples for each input row."""
        samples = pd.DataFrame({
            "_input_row_id": [0, 0, 0],
            "income": [50000, 50000, 50000],  # No variance
        })
        input_mask = pd.DataFrame({"income": [False]})

        result = ImputationResult(samples=samples, input_mask=input_mask, n_samples=3)
        stds = result.std()

        assert stds.loc[0, "income"] == 0.0

    def test_quantile(self):
        """quantile() should compute quantile across samples."""
        samples = pd.DataFrame({
            "_input_row_id": [0] * 5,
            "income": [10000, 20000, 30000, 40000, 50000],
        })
        input_mask = pd.DataFrame({"income": [False]})

        result = ImputationResult(samples=samples, input_mask=input_mask, n_samples=5)
        median = result.quantile(0.5)

        assert median.loc[0, "income"] == 30000

    def test_quantile_bounds(self):
        """quantile(0) and quantile(1) should return min and max."""
        samples = pd.DataFrame({
            "_input_row_id": [0, 0, 0],
            "value": [100, 200, 300],
        })
        input_mask = pd.DataFrame({"value": [False]})

        result = ImputationResult(samples=samples, input_mask=input_mask, n_samples=3)

        assert result.quantile(0.0).loc[0, "value"] == 100
        assert result.quantile(1.0).loc[0, "value"] == 300


class TestBaseSynthesisModel:
    """Tests for BaseSynthesisModel abstract base class."""

    def test_cannot_instantiate_directly(self):
        """BaseSynthesisModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSynthesisModel()

    def test_subclass_must_implement_abstract_methods(self):
        """Subclass must implement all abstract methods."""

        class IncompleteModel(BaseSynthesisModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_complete_subclass_can_be_instantiated(self):
        """Complete subclass that implements all methods can be instantiated."""

        class CompleteModel(BaseSynthesisModel):
            def fit(self, data, mask=None, **kwargs):
                return self

            def generate(self, n, **kwargs):
                persons = pd.DataFrame({"person_id": range(n), "value": np.random.randn(n)})
                return SyntheticPopulation(persons=persons)

            def impute(self, partial_obs, n_samples=100, **kwargs):
                n = len(partial_obs)
                samples = pd.DataFrame({
                    "_input_row_id": np.repeat(range(n), n_samples),
                    "value": np.random.randn(n * n_samples),
                })
                return ImputationResult(
                    samples=samples,
                    input_mask=pd.DataFrame({"value": [False] * n}),
                    n_samples=n_samples,
                )

            def log_prob(self, data, mask=None):
                return np.zeros(len(data))

        model = CompleteModel()
        assert isinstance(model, BaseSynthesisModel)

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""

        class ChainableModel(BaseSynthesisModel):
            def fit(self, data, mask=None, **kwargs):
                self.fitted = True
                return self

            def generate(self, n, **kwargs):
                return SyntheticPopulation(persons=pd.DataFrame({"id": range(n)}))

            def impute(self, partial_obs, n_samples=100, **kwargs):
                return ImputationResult(
                    samples=pd.DataFrame(),
                    input_mask=pd.DataFrame(),
                    n_samples=n_samples,
                )

            def log_prob(self, data, mask=None):
                return np.zeros(len(data))

        model = ChainableModel()
        data = pd.DataFrame({"x": [1, 2, 3]})

        returned = model.fit(data)

        assert returned is model
        assert model.fitted is True


class TestBaseTrajectoryModel:
    """Tests for BaseTrajectoryModel abstract base class."""

    def test_cannot_instantiate_directly(self):
        """BaseTrajectoryModel is abstract."""
        with pytest.raises(TypeError):
            BaseTrajectoryModel()

    def test_generate_calls_generate_trajectories(self):
        """Default generate() should call generate_trajectories()."""

        class TrajectoryModel(BaseTrajectoryModel):
            def __init__(self):
                self.generate_traj_called = False

            def fit(self, data, mask=None, **kwargs):
                return self

            def generate_trajectories(self, n, T, **kwargs):
                self.generate_traj_called = True
                self.last_n = n
                self.last_T = T
                persons = pd.DataFrame({
                    "person_id": np.repeat(range(n), T),
                    "period": np.tile(range(T), n),
                })
                return SyntheticPopulation(persons=persons)

            def impute(self, partial_obs, n_samples=100, **kwargs):
                return ImputationResult(pd.DataFrame(), pd.DataFrame(), n_samples)

            def log_prob(self, data, mask=None):
                return np.zeros(len(data))

        model = TrajectoryModel()
        result = model.generate(n=10, T=5)

        assert model.generate_traj_called
        assert model.last_n == 10
        assert model.last_T == 5
        assert isinstance(result, SyntheticPopulation)

    def test_generate_default_T(self):
        """generate() should default to T=1 if not specified."""

        class TrajectoryModel(BaseTrajectoryModel):
            def fit(self, data, mask=None, **kwargs):
                return self

            def generate_trajectories(self, n, T, **kwargs):
                self.last_T = T
                return SyntheticPopulation(persons=pd.DataFrame({"id": range(n)}))

            def impute(self, partial_obs, n_samples=100, **kwargs):
                return ImputationResult(pd.DataFrame(), pd.DataFrame(), n_samples)

            def log_prob(self, data, mask=None):
                return np.zeros(len(data))

        model = TrajectoryModel()
        model.generate(n=10)  # No T specified

        assert model.last_T == 1


class TestBaseGraphModel:
    """Tests for BaseGraphModel abstract base class."""

    def test_cannot_instantiate_directly(self):
        """BaseGraphModel is abstract."""
        with pytest.raises(TypeError):
            BaseGraphModel()

    def test_inherits_from_trajectory_model(self):
        """BaseGraphModel should inherit from BaseTrajectoryModel."""
        assert issubclass(BaseGraphModel, BaseTrajectoryModel)

    def test_requires_generate_population(self):
        """Subclass must implement generate_population."""

        class IncompleteGraph(BaseGraphModel):
            def fit(self, data, mask=None, **kwargs):
                return self

            def generate_trajectories(self, n, T, **kwargs):
                return SyntheticPopulation(persons=pd.DataFrame())

            def impute(self, partial_obs, n_samples=100, **kwargs):
                return ImputationResult(pd.DataFrame(), pd.DataFrame(), n_samples)

            def log_prob(self, data, mask=None):
                return np.zeros(len(data))

            # Missing generate_population and sample_events

        with pytest.raises(TypeError):
            IncompleteGraph()

    def test_complete_graph_model(self):
        """Complete GraphModel can be instantiated."""

        class CompleteGraph(BaseGraphModel):
            def fit(self, data, mask=None, **kwargs):
                return self

            def generate_trajectories(self, n, T, **kwargs):
                return SyntheticPopulation(persons=pd.DataFrame({"id": range(n)}))

            def generate_population(self, n_households, T, **kwargs):
                persons = pd.DataFrame({
                    "person_id": range(n_households * 2),
                    "household_id": np.repeat(range(n_households), 2),
                })
                households = pd.DataFrame({
                    "household_id": range(n_households),
                })
                return SyntheticPopulation(persons=persons, households=households)

            def sample_events(self, population, period):
                return {}  # No events

            def impute(self, partial_obs, n_samples=100, **kwargs):
                return ImputationResult(pd.DataFrame(), pd.DataFrame(), n_samples)

            def log_prob(self, data, mask=None):
                return np.zeros(len(data))

        model = CompleteGraph()
        pop = model.generate_population(n_households=5, T=3)

        assert isinstance(pop, SyntheticPopulation)
        assert pop.n_households == 5


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_interface_exists(self):
        """save() and load() methods should exist on base class."""
        # BaseSynthesisModel has save/load methods
        assert hasattr(BaseSynthesisModel, "save")
        assert hasattr(BaseSynthesisModel, "load")

    def test_save_requires_path(self):
        """save() requires a path argument."""
        import inspect
        sig = inspect.signature(BaseSynthesisModel.save)
        params = list(sig.parameters.keys())
        assert "path" in params
