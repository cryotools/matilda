# Testing and numerical impact reports

MATILDA's characterization suite protects the complete numerical result of a
standard 2000–2020 glacier-evolution simulation. It compares the compact and
full daily process tables, KGE, summary statistics, glacier elevation
distribution, and annual glacier evolution. Dates, column order, data types,
and missing-value positions are part of the reference contract.

## Local Python 3.11 environment

The maintained characterization environment uses Python 3.11.15. Create an
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

The command writes three files:

- `variable_impact.csv` reports the number of changed values, maximum absolute
  change, mean absolute change, RMSE, changes in totals and means, and the first
  affected index for every output variable.
- `annual_impact.csv` reports pointwise, mean, and total changes by calendar
  year for outputs with date indexes.
- `summary.json` lists the affected model-process groups.

These reports are intended to show which processes, variables, and periods are
affected by a change. They can therefore guide decisions about which figures,
tables, calibrations, or catchment simulations need to be repeated.

## Reference policy

The reference file has a recorded checksum and must not be replaced merely to
make a changed test pass. When an intentional scientific correction changes
the output, retain its impact reports for review first. Update the reference
only after the scientific consequences and compatibility implications have
been accepted.

The current reference covers one long glacier-evolution scenario. Separate
tests are still required before changing zero-glacier, full-glacier,
fixed-elevation, short-period, and calibration pathways.

The continuous-integration workflow also builds a wheel and verifies that its
package files are byte-for-byte copies of the maintained sources.
