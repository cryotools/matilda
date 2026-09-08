# Testing and numerical impact reports

MATILDA's model output consistency tests protect the complete numerical result
of a standard 2000–2020 glacier-evolution simulation. It compares the compact
and full daily process tables, KGE, summary statistics, glacier elevation
distribution, and annual glacier evolution. Dates, column order, data types,
and missing-value positions are part of the reference contract.

## Local Python 3.11 environment

The maintained test environment uses Python 3.11.15. Create an
isolated environment from that interpreter and install the pinned test tools:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
```

Run all tests with:

```bash
.venv/bin/python -m pytest
```

The test requirements pin the direct and secondary dependencies used for the
numerical reference. The `.venv/` directory is local and is excluded from
version control.

## Quantifying the effect of model changes

To compare the working implementation with the maintained reference and write
detailed reports, run:

```bash
.venv/bin/python -m pytest --impact-report=impact-report
```

The command writes three files for the long glacier-evolution reference:

- `variable_impact.csv` reports the number of changed values, maximum absolute
  change, mean absolute change, RMSE, changes in totals and means, and the first
  affected index for every output variable.
- `annual_impact.csv` reports pointwise, mean, and total changes by calendar
  year for outputs with date indexes.
- `summary.json` lists the affected model-process groups.

The `zero_glacier/` and `fixed_glacier/` subdirectories contain the same three
reports for the exact synthetic model-mode references.

These reports are intended to show which processes, variables, and periods are
affected by a change. They can therefore guide decisions about which figures,
tables, calibrations, or catchment simulations need to be repeated.

## Synthetic model-mode tests

Deterministic synthetic forcing covers a complete setup year followed by the
2000–2001 simulation period. This includes winter and summer conditions and the
leap day in 2000 without requiring observed runoff from another catchment.

The synthetic tests cover:

- parameter initialization and validation;
- rain–snow partitioning and precipitation conservation;
- snow and ice melt limits, refreezing, and glacier-reservoir bookkeeping;
- zero-glacier and fixed-glacier public simulation paths;
- non-negative stores and fluxes where physically required;
- calendar completeness, deterministic repetition, and input immutability;
- glacier loss during the first annual geometry update;
- one-day simulation output and the documented saved-file set;
- the main public call signatures;
- repeatable model evaluations initiated through `mspot`, including seeded
  glacier-only sampling.

The runoff series used by the `mspot` test is synthetic and non-constant. Its
only purpose is to exercise unit conversion, alignment, and objective-function
calculation. It is not used to assess predictive performance or scientific
validity.

## Reference policy

The reference file has a recorded checksum and must not be replaced merely to
make a changed test pass. When an intentional scientific correction changes
the output, retain its impact reports for review first. Update the reference
only after the scientific consequences and compatibility implications have
been accepted.

Exact numerical references cover the long glacier-evolution scenario and the
synthetic zero-glacier and fixed-elevation scenarios. Reference files have
recorded checksums and are only regenerated after an intentional scientific
change has been assessed and accepted. The explicit command for that step is:

```bash
.venv/bin/python -m tests.generate_synthetic_references --replace
```

Separate tests are still required for positive cumulative mass-balance
handling and serial-versus-parallel calibration equivalence.

The current SPOTPY parallel route uses MPI. Parallel-equivalence checks are not
part of the local test suite because its pinned environment does not include an
MPI runtime.

The continuous-integration workflow also builds a wheel and verifies that its
package files are byte-for-byte copies of the maintained sources.
