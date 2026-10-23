# CLI placement repair completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive one-line runner repair, real constructor regression and exact validation/ownership boundaries. |

TaskID:**D29-CLI-PLACEMENT-REPAIR**. Base1be91bad431d3bd886de7f5d80c19bd7e6c54a25;branchfeat/step5-cli-placement-repair;isolatedworktree `/data/ycfeng/Frontier/.worktrees/step5-cli-placement-repair`.

## Delivered behavior

- `tests/integration/run_step5_measured_main.py`: position-onlycommandbranch now emits existingglobal `--collective_sim_cc_backend_config_placement_order TP,DP` next toprofiling_data_dir. FixedTP8/DP2fixture selects existing approvedfiniteplacement. No globaldefault/newflag/roleoverride/backend/gate/numerical/timing changes.
- `tests/integration/test_step5_measured_main.py`: offline/online x P/D regression uses actualrunner.command->SimulationConfigCLIparser->realCluster/lazyfactory->exactexistingCPUwriterplan. Checksoriginaldimensions,fullkey,cost0.03msandcompletecountreceipt;noexceptions-as-GREEN. Legacybalancedcommandretainsolddefault. Persistenttestimports onlycurrentrepo andwrites onlytmp_path.

## Observed evidence

- **RED4FAIL2.32s** at originalplacementguardbeforefix.
- **GREEN5PASS2.18s** realregression+existingcommandcheck.
- **Finalwholefiles57PASS/6SKIP59.24s**,noFAIL/ERROR/exclusions. SkipsareunchangedBF16/deferredprepareandmissingtorchcases;no skipadded.
- **Retainedfixture03short-chain**: currentrunner->realconfig/Cluster/backend,offline/onlineP/Dallcost0.03ms/exactkeys. Manifestandcollectivehashesunchanged;metrics/cacheoutputrelocatedbeforeCLIparse.
- **Budget/AST/whitespacePASS**:two paths+40/-0=40/65,driver47/66,test260/283;donorunchanged70cap;CORE3873/4404,global9116/10000,111pathsunchanged.

Exactdiff:placement_repair.diff;machinebudget:budget.json;commands,hashesandlimitations:test_report_2026-10-23_cli_placement.md,inthisdirectory.

## Remaining boundaries

No unresolvedownedrepairfailure. Rootindependentreview,integration/freeze,T2sourcealignmentandfulloffline/online6144/1024replaywith2400sremainpending/outsideagentwork. No newfullCLI/unit/GPU/networkcalls. Integrated04,retainedruns,inputfixture03androotgitlinkuntouched. Previouslydisclosedscoutoverwriteofmain05offlineconfigisnotoriginalevidence;notrestored/relabelled. Inertrole-scopedplacementoverrideisdeferred,notfixed.
