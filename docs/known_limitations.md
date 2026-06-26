# Known Limitations

1. **Synthetic benchmark boundary â€” High.** IBM AML results are controlled
   benchmark evidence, not production-bank validation or regulatory approval.

2. **No hosted Medium-data execution â€” Medium.** The GitHub workflow runs unit
   and static-quality checks; it does not execute the multi-million-row
   HI-Medium or LI-Medium runs.

3. **Locked configuration evidence â€” Medium.** v0.9.4 evaluates one
   no-bank-identity configuration and documented seed. It does not establish
   universal performance across institutions, data-generating regimes, or model
   families.

4. **Secondary calendar-time diagnostic â€” Medium.** Both Medium runs report a
   6-hour diagnostic lower bound of 0.00%. This analysis is explicitly
   secondary; the pre-registered primary rule is transaction-mass inference.

5. **No production operating model â€” High.** Case management, analyst feedback,
   drift monitoring, access control, retention policy, governance, and incident
   response are out of scope.

6. **External artifacts â€” Low.** Medium inputs, trained models, predictions,
   and local evidence are untracked. Reproduction needs external data and
   suitable compute.

## Legacy boundary
The v0.8.2 release-security overlay remains a supporting component. Its
historical qualification does not qualify the combined AML platform.