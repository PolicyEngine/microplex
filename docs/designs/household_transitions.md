# Household Transition Modeling for Panel Synthesis

## Overview

Dynamic microsimulation models treat **individuals as the primary unit**, with households emerging from transition events (marriage, divorce, leaving home, etc.). This design extends `MultiSourceFusion` to support household dynamics in panel data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INDIVIDUAL-CENTRIC PANEL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER 1: Individual Trajectories (current MultiSourceFusion)   │
│  ─────────────────────────────────────────────────────────────  │
│  - Demographics: age, sex, education                             │
│  - Economics: income, employment, job composition                │
│  - Cross-survey imputation (SIPP ↔ CPS ↔ SCF)                   │
│                                                                  │
│  LAYER 2: Household State (NEW)                                  │
│  ─────────────────────────────────────────────────────────────  │
│  - household_id[t]: which household at time t                    │
│  - relationship[t]: role in household (head, spouse, child...)   │
│  - hh_size[t]: current household size                            │
│  - hh_income[t]: sum of member incomes (derived)                 │
│                                                                  │
│  LAYER 3: Transition Events (NEW)                                │
│  ─────────────────────────────────────────────────────────────  │
│  Events that change household membership:                        │
│  - MARRIAGE: person joins spouse's HH or forms new HH            │
│  - DIVORCE: person leaves HH, may form new HH                    │
│  - LEAVE_HOME: child/young adult leaves parental HH              │
│  - JOIN_HH: person joins existing HH (roommate, partner)         │
│  - DEATH: person exits, HH may dissolve                          │
│  - BIRTH: new person added to HH                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Requirements

### SIPP Panel Structure
```
person_id     | period | household_id | relationship | marital_status | age | income
──────────────┼────────┼──────────────┼──────────────┼────────────────┼─────┼────────
P001          | 1      | H001         | head         | married        | 35  | 5000
P001          | 2      | H001         | head         | married        | 35  | 5200
...
P001          | 24     | H001         | head         | divorced       | 37  | 4800  # DIVORCE
P001          | 25     | H002         | head         | divorced       | 37  | 4900  # New HH
```

### Derived Transition Events
```python
def extract_transitions(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Extract household transition events from panel data."""

    events = []
    for person_id, group in panel_df.groupby('person_id'):
        group = group.sort_values('period')

        for i in range(1, len(group)):
            prev, curr = group.iloc[i-1], group.iloc[i]

            # Detect household change
            if prev['household_id'] != curr['household_id']:
                event_type = classify_transition(prev, curr)
                events.append({
                    'person_id': person_id,
                    'period': curr['period'],
                    'event_type': event_type,
                    'from_hh': prev['household_id'],
                    'to_hh': curr['household_id'],
                    # Features for modeling
                    'age': curr['age'],
                    'marital_status_before': prev['marital_status'],
                    'marital_status_after': curr['marital_status'],
                    'income': prev['income'],
                    'hh_size_before': prev['hh_size'],
                })

    return pd.DataFrame(events)

def classify_transition(prev, curr) -> str:
    """Classify transition event type from state changes."""

    ms_before, ms_after = prev['marital_status'], curr['marital_status']
    rel_before, rel_after = prev['relationship'], curr['relationship']

    # Marriage: single/divorced → married, new HH or join spouse's
    if ms_before in ('single', 'divorced') and ms_after == 'married':
        return 'MARRIAGE'

    # Divorce: married → divorced/separated, leave HH
    if ms_before == 'married' and ms_after in ('divorced', 'separated'):
        return 'DIVORCE'

    # Leave home: child → head of new HH
    if rel_before == 'child' and rel_after == 'head':
        return 'LEAVE_HOME'

    # Join household: head → non-head (moving in with someone)
    if rel_before == 'head' and rel_after != 'head':
        return 'JOIN_HH'

    return 'OTHER'
```

## Transition Probability Models

### Event Probability Model
```python
class HouseholdTransitionModel:
    """Models P(event | person_features, household_features, time)."""

    def __init__(self):
        self.event_models = {}  # One model per event type

    def fit(self, panel_df: pd.DataFrame, events_df: pd.DataFrame):
        """Train transition probability models."""

        # For each event type, train a binary classifier
        for event_type in ['MARRIAGE', 'DIVORCE', 'LEAVE_HOME', 'JOIN_HH']:

            # Features: age, income, marital_status, hh_size, relationship, duration
            X = self._build_features(panel_df, event_type)
            y = self._build_targets(panel_df, events_df, event_type)

            # Use logistic regression or gradient boosting
            self.event_models[event_type] = LogisticRegression()
            self.event_models[event_type].fit(X, y)

    def predict_event_probs(self, person_state: dict) -> dict:
        """Predict probability of each event type for a person-period."""

        probs = {}
        for event_type, model in self.event_models.items():
            X = self._state_to_features(person_state)
            probs[event_type] = model.predict_proba(X)[0, 1]

        return probs
```

### Spouse Matching (for Marriage Events)
```python
class SpouseMatcher:
    """Match individuals for marriage events."""

    def __init__(self, strategy: str = 'propensity'):
        self.strategy = strategy
        self.match_model = None

    def fit(self, couples_df: pd.DataFrame):
        """Learn spouse matching patterns from observed couples."""

        # Features: age_diff, income_ratio, education_match, location
        # Target: observed couple (positive) vs random pair (negative)
        ...

    def find_match(self, person: dict, candidates: pd.DataFrame) -> int:
        """Find best spouse match from candidate pool."""

        if self.strategy == 'propensity':
            # Score all candidates, sample weighted by score
            scores = self.match_model.predict_proba(
                self._build_pair_features(person, candidates)
            )[:, 1]
            return np.random.choice(candidates.index, p=scores/scores.sum())

        elif self.strategy == 'nearest':
            # Find nearest neighbor in feature space
            ...
```

## Generation with Household Dynamics

```python
class PanelSynthesizerWithHouseholds:
    """Generate synthetic panel with household transitions."""

    def __init__(
        self,
        individual_model: MultiSourceFusion,
        transition_model: HouseholdTransitionModel,
        spouse_matcher: SpouseMatcher,
    ):
        self.individual_model = individual_model
        self.transition_model = transition_model
        self.spouse_matcher = spouse_matcher

    def generate(
        self,
        n_persons: int,
        n_periods: int,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic panel with evolving household structure.

        Returns:
            persons_df: Person-period level data
            households_df: Household-period level data (derived)
        """

        rng = np.random.default_rng(seed)

        # Step 1: Generate initial states (t=0)
        # Includes household_id, relationship from initial distribution
        persons_t0 = self._generate_initial_state(n_persons, rng)

        # Step 2: Evolve forward through time
        all_records = [persons_t0]
        current_state = persons_t0.copy()
        next_hh_id = current_state['household_id'].max() + 1

        for t in range(1, n_periods):
            new_state = []

            for _, person in current_state.iterrows():
                # Get transition probabilities
                event_probs = self.transition_model.predict_event_probs(person)

                # Sample event (or no event)
                event = self._sample_event(event_probs, rng)

                # Apply event effects
                if event == 'MARRIAGE':
                    spouse = self.spouse_matcher.find_match(
                        person,
                        current_state[current_state['marital_status'] == 'single']
                    )
                    # Update both person and spouse
                    person, spouse, next_hh_id = self._apply_marriage(
                        person, spouse, next_hh_id, rng
                    )

                elif event == 'DIVORCE':
                    person, next_hh_id = self._apply_divorce(
                        person, next_hh_id, rng
                    )

                elif event == 'LEAVE_HOME':
                    person, next_hh_id = self._apply_leave_home(
                        person, next_hh_id, rng
                    )

                # Evolve individual attributes (income, etc.)
                person = self.individual_model.evolve_person(person, t)
                new_state.append(person)

            current_state = pd.DataFrame(new_state)
            current_state['period'] = t
            all_records.append(current_state)

        persons_df = pd.concat(all_records, ignore_index=True)
        households_df = self._derive_households(persons_df)

        return persons_df, households_df

    def _derive_households(self, persons_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate person data to household level."""

        return persons_df.groupby(['household_id', 'period']).agg({
            'person_id': 'count',  # n_persons
            'income': 'sum',       # hh_income
            'age': 'min',          # age of youngest
            # ... other aggregations
        }).reset_index()
```

## Integration with MultiSourceFusion

```python
class HierarchicalMultiSourceFusion:
    """Multi-source fusion with household hierarchy support."""

    def __init__(
        self,
        shared_vars: List[str],
        all_vars: List[str],
        household_vars: List[str],  # NEW: HH-level vars
        n_periods: int = 6,
    ):
        # Individual-level fusion (existing)
        self.individual_fusion = MultiSourceFusion(
            shared_vars=shared_vars,
            all_vars=all_vars,
            n_periods=n_periods,
        )

        # Household-level components (new)
        self.household_vars = household_vars
        self.transition_model = HouseholdTransitionModel()
        self.spouse_matcher = SpouseMatcher()

    def add_source(
        self,
        name: str,
        person_data: pd.DataFrame,
        household_data: pd.DataFrame,  # NEW
        source_vars: List[str],
        household_id_col: str = 'household_id',
    ):
        """Add a survey source with person and household data."""

        # Link person to household features
        person_with_hh = person_data.merge(
            household_data,
            on=household_id_col
        )

        self.individual_fusion.add_source(name, person_with_hh, source_vars)

        # Extract transition events if panel data
        if self._is_panel(person_data):
            events = extract_transitions(person_data)
            self.transition_model.fit_partial(person_data, events)

    def generate(
        self,
        n_persons: int,
        n_periods: int,
        include_transitions: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate with optional household dynamics."""

        if include_transitions:
            return PanelSynthesizerWithHouseholds(
                self.individual_fusion,
                self.transition_model,
                self.spouse_matcher,
            ).generate(n_persons, n_periods)
        else:
            # Static households (current behavior)
            persons = self.individual_fusion.generate(n_persons)
            households = self._derive_households(persons)
            return persons, households
```

## Calibration Targets

Household transition models should be calibrated to match:

1. **Marriage rates** by age, sex, education
2. **Divorce rates** by duration, children, income
3. **Household size distribution** by geography
4. **Living arrangement patterns** (alone, with parents, with roommates)

Sources:
- Census/ACS household composition
- Vital statistics (marriage/divorce rates)
- SIPP panel transition rates (when available)

## Current Limitations

1. **SIPP data only has one wave per panel** - no observed transitions
2. **CPS is cross-sectional** - household is static
3. **Need full 4-year SIPP panels** for proper transition modeling

## Next Steps

1. [ ] Acquire full SIPP longitudinal files (all waves per panel)
2. [ ] Implement `extract_transitions()` function
3. [ ] Train logistic models for each event type
4. [ ] Implement spouse matching from observed couples
5. [ ] Integrate with `MultiSourceFusion.generate()`
6. [ ] Add calibration to match aggregate transition rates
