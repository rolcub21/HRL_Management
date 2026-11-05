# Conditioned-VCG final claim evidence tracker

Last implementation audit: 2026-11-05.

This file is the authoritative status list for the final 5x5 and 8x8
conditioned-VCG evidence.  A result is `VERIFIED COMPLETE` only when all of the
following have been checked in the implementation and artifacts, not merely
reported in a terminal or README:

1. the executed scenario, instance manifest, model seeds, selected checkpoint
   hashes, and method arms match the claimed comparison;
2. the behavioral delta has been inspected in code and no additional controller
   change is mixed into the ablation;
3. the expected rows exist, authenticate, and the report contains no missing or
   reused-incompatible coordinate;
4. strict completion and the components of J are audited under the same
   environment and objective;
5. frozen evaluations contain no training and do not mutate checkpoints;
6. the written claim is no stronger than the implemented endpoint.

Status vocabulary:

- `VERIFIED COMPLETE`: implementation and complete local artifacts checked.
- `REPORTED COMPLETE`: outcome reported, but the final artifacts have not been
  authenticated in this checkout.
- `PARTIAL`: some rows or checkpoints exist, but the declared endpoint is not
  complete.
- `PREPARED`: implementation, hashes, and required live integration gate pass,
  but the declared evaluation rows have not been executed.
- `MISSING`: the required comparison has not been implemented or executed.
- `OPTIONAL`: not required unless it replaces the frozen primary method.

## Frozen method identity

- 5x5 Stage A: E195, frozen by E196.  Selected episodes are 300, 300, and 160
  for model seeds 0, 1, and 2.
- 8x8 Stage A: E215 seed 0 and E218 seeds 1 and 2, evaluated together by E222.
- Objective: `2*T + E + 0.5*W + 0.1*L + 10*R`, gamma 1.
- Certification: exact closed-admission strict-macro recoverability inherited
  from E191/E78.  This is not the E181/E184 transition-access/drive-under
  branch.
- Progress guard: E188 mask-only guard, threshold two non-progress
  Reconfigures; retain Accept, Deliver, and Defer when available; otherwise use
  one retained witness step and recheck.
- Defer: at most 10 primitive steps per macro and 16 consecutive Defer macros.
- Primary deployment dial: immediate-only rehandle repricing.  Learned
  future-rehandle heads are ablations unless a later experiment passes the
  multiseed replacement gate.

## Claim 1: certification keeps every accepted block retrievable

| Test | Size | Audited status | Implementation-qualified conclusion |
| --- | --- | --- | --- |
| E199 certification stress | 5x5 | VERIFIED COMPLETE | Frozen E195 seed 0, VCG certified versus physical-unfiltered, 15 development stress cases at nominal 80%, 100%, and 125%. No non-safe candidates appeared, so both arms were identical. It does **not** replace E233. |
| E200 dense unsafe-choice audit | 5x5 | VERIFIED COMPLETE | At the predeclared 7/8-occupied boundary, the same critic selected an UNSAFE first action in all 15 unfiltered cases. Certified VCG completed 15/15; unfiltered VCG completed 0/15. Appropriate illustrated mechanism case, not a workload benchmark. |
| E198 certification-free training, seed 0 | 5x5 | VERIFIED COMPLETE | Full 300-episode training with exact labels demoted to shadow telemetry. Supporting evidence only: it changes the learned policy and therefore does not hold ranking fixed. |
| E233 four-arm causal ablation | 8x8 | PARTIAL; RUNNING IN USER TERMINAL | Gate passed. There are 120 referenced rows plus 143/230 completed new rows (263/350 final rows). The VCG physical-unfiltered panel and monitored plain-Dynamic panel are complete; certified Dynamic has 23/30 panel rows and is currently in its next coordinate. The seed-0 physical-unfiltered panel had five failures, all at the 80% workload; every failure was `stranded_block` and followed an exhaustively UNSAFE Accept recorded by the non-authoritative monitor. Seven panel rows and all 80 stress rows remain. Final multiseed/stress conclusions remain pending. The sandboxed process table cannot see this user-terminal process, so status is tracked from its coordinate log and artifacts. |
| E237 matched four-arm causal ablation | 5x5 | VERIFIED COMPLETE; DYNAMIC CAUSAL ARM QUALIFIED | All 350/350 rows authenticate and all arms complete every case. VCG certified and physical-unfiltered are identical on all 90 panel and 30 stress pairs; no exhaustive-UNSAFE entry occurs, so certification was not empirically needed on these workloads. Certified Dynamic is about 11 J worse than plain Dynamic despite zero unsafe entries, proving the macro projection is not an exact no-filter behavioral match to the primitive scheduler. Do not use that pair for a fixed-ranker causal claim. E200 remains the positive dense illustrated failure case. |
| E238 matched-ranker prefilled curve | 5x5 | VERIFIED COMPLETE | All 80/80 rows authenticate; all 20 prefilled starts at 4/8, 5/8, 6/8, and 7/8 occupancy are exact SAFE with nonempty completion witnesses. The certified VCG and Dynamic arms completed 20/20 each. Physical-unfiltered VCG completed 7/20: 5/5, 2/5, 0/5, and 0/5 as fill rose from 4/8 to 7/8. Physical-unfiltered Dynamic completed 17/20: 5/5, 4/5, 5/5, and 3/5. Every one of the 16 failures was `stranded_block` and followed a monitor-confirmed exhaustive-UNSAFE Accept. Both Dynamic arms instantiate one authenticated ranker (`b5baf5...e131`) on the same macro interface; their only behavioral delta is SAFE-only versus physical frontier exposure. This is the valid replacement for E237's non-identical Dynamic pair and the primary 5x5 mechanism evidence. |
| E243 expanded E238-design replication | 5x5 | PREPARED; REMOTE EXECUTION PENDING | The implementation freezes 25 witnessed-safe days per fill level (4/8, 5/8, 6/8, 7/8), for 100 starts and 400 episodes across the exact four E238 arms. All 100 starts passed the local exact-witness preflight; the four-route live integration gate passed; the one-ranker Dynamic construction is content-addressed. The frozen E195 seed-0 source bundle is authenticated separately from Git. No evaluation rows have been run yet. |

The paper must not describe E199 as the 5x5 counterpart of E233.

## Claim 2: learned control outperforms matched heuristics

| Test | Size | Audited status | Implementation-qualified conclusion |
| --- | --- | --- | --- |
| E195/E196 Stage A and freeze | 5x5 | VERIFIED COMPLETE | Three 300-episode training runs; immutable selected checkpoints at episodes 300, 300, and 160. |
| E201 matched test panel | 5x5 | VERIFIED COMPLETE | Complete 240-row panel: three VCG seeds, Dynamic, and four GA nuisance replicates on 30 locked instances. |
| E215/E218 Stage A and freeze | 8x8 | VERIFIED THROUGH E222 EXECUTION | E222 authenticated and executed all three selected Stage-A models. Preserve the original freeze manifests and hashes with the paper artifacts. |
| E222 matched test panel | 8x8 | VERIFIED COMPLETE | 240/240 rows, no training runs. VCG completed 90/90. Pooled VCG J is 2484.15 versus Dynamic 3657.47, a 32.1% reduction. GA completed 29/30, so unconditional mean-J comparison to GA is not valid. |

## Claim 3: the progress guard is needed

| Test | Size | Audited status | Implementation-qualified conclusion |
| --- | --- | --- | --- |
| E225 frozen guard-removal replay | 8x8 | VERIFIED COMPLETE | Same frozen seed-0 critic and certified frontier, with all guard intervention disabled while retaining its trained context inputs. Strict completion fell to 7/9 and mean J rose from 2452.63 to 108971.4. |
| E226 guard-free training | 8x8 | VERIFIED STOP-RULE OUTCOME | Paused at episode 58, not a completed 300-episode run. The evidence is the predeclared early-stop trajectory with large greedy rehandle loops, not a final checkpoint claim. |
| E228 mechanism audit | 8x8 | COMPLETE ARTIFACTS PRESENT; REPORT CLAIM REQUIRES SEPARATE AUDIT | Diagnoses minimum-selection underestimation on E226 states. Do not merge its mechanistic conclusion into the guard claim without checking the report. |
| E229 multistep/greedy repair | 8x8 | VERIFIED STOPPED FUTURE WORK | Paused at episode 10. It is not evidence that the guard can be removed. |

Identifier correction: in the current repository E225 is the guard-removal
experiment. It is **not** the seed-1 Stage-B divergence diagnostic.

## Claim 4: deployment-time rehandle control

| Test | Size | Audited status | Implementation-qualified conclusion |
| --- | --- | --- | --- |
| E210/E211 learned and immediate-only dials | 5x5 | VERIFIED COMPLETE, DEVELOPMENT ONLY | Three-seed development ablation. E211's learned future term beat immediate-only on only 18.5% of paired comparisons, so it is not the primary dial. |
| E214 learned dial | 8x8 | VERIFIED COMPLETE, SEED 0 DEVELOPMENT | Supports the learned-head mechanism on one seed only. |
| E223 multiseed learned dial | 8x8 | REPORTED COMPLETE; LOCAL FINAL ARTIFACTS MISSING | The learned future-rehandle head remains an ablation, not the primary dial, but its cross-seed instability is retained for one explanatory paragraph. Transfer and authenticate the E221/E223 fitted heads, rows, and final report before citation. |
| E241 seed-1 first-divergence diagnostic | 8x8 | IMPLEMENTED; SOURCE IMPORT AND RUN PENDING | Unique development-only diagnostic covering all six seed-1 E223 learned-versus-immediate pairs (20%, 50%, and 80% at lambda 20 and 40). It authenticates and imports the exact E221 endpoint, E218 Stage-A identity, E223 manifest/report, and paired rows; then it replays only the shared prefix and computes the exact immediate-only shadow choice on the same certified frontier. The first divergence is joined to the authenticated full-episode outcomes. This is a mechanism diagnostic, not an isolated causal intervention and not promotion of the learned dial. Focused tests pass, but the live import/authentication gate cannot run in this checkout until the missing E221/E223 artifacts are supplied. |
| E217 offline fixed-policy head | 8x8 | VERIFIED COMPLETE | Final verdict `keep_e214`; the lighter offline head did not replace the on-policy learned procedure. |
| E231 simple high-lambda dial | 8x8 | VERIFIED COMPLETE, DEVELOPMENT ONLY | Immediate-only high-lambda extension; no dial training required. |
| E235 full simple-dial test-panel curve | 8x8 | PARTIAL; RUNNING IN USER TERMINAL | Declared lambdas 10, 20, 40, 80, 160 over all three seeds. All 180 new lambda-20 and lambda-40 rows are complete. Lambda 160 seed 0 has 12/30 rows, giving 192/270 new rows overall; lambda-160 seeds 1--2 remain. The current coordinate is seed 0 at 50% workload. Lambda-10 and lambda-80 rows are authenticated reuse. The sandboxed process table cannot see this user-terminal process. |
| E239 immediate-only simple-dial curve on E201 instances | 5x5 | COMPLETE | All 360 new frozen rows at lambda 20, 40, 80, and 160 completed, with 90 authenticated lambda-10 rows reused and strict completion 1.0 throughout. Rehandles fall to zero by lambda 80 at every workload. The visible cost is timing: on 80% workload, mean J10 rises from 117.6 at lambda 10 to 195.3 at lambda 80/160 while mean rehandles fall from 1.5 to 0. The curve saturates at lambda 80 on this panel. No learned future-rehandle term and no training are present. |
| E236 counterfactual alternative-placement Stage B | 8x8 | VERIFIED TRAINING ARTIFACT; REJECTED END TO END BY E242 | Seed 0 completed all 72/72 counterfactual branches with strict completion 1.0 and fitted the declared small head. Its head-only admission-30 ranking screen failed; E242 then bypassed that proxy and evaluated the complete frozen controller on realized rehandles. The behavioral result was also negative, so seeds 1--2 are unnecessary. |
| E242 E236 end-to-end rehandle evaluation | 8x8 | VERIFIED COMPLETE, DEVELOPMENT ONLY | All 15/15 frozen episodes completed strictly. Mean Stage-A rehandles were 10.667. At lambda 20, immediate-only averaged 11.0 and learned Stage B 18.0; at lambda 40 they averaged 8.333 and 20.0. Learned Stage B had more rehandles than both Stage A and immediate-only in all 6/6 paired comparisons. The future head is rejected; immediate-only remains the deployment method. |

## Claim 5: exact acceleration is affordable and decision preserving

| Test | Size | Audited status | Required endpoint |
| --- | --- | --- | --- |
| Final-model acceleration ablation (E240) | 8x8 | VERIFIED COMPLETE; PAPER EVIDENCE | All equivalence checks passed across 36 decisions and 2,780 candidate evaluations on the 20%, 50%, and 80% development workloads. Candidate identity/order, certificate verdicts and replay-valid witnesses, graph inputs, Q values for all three selected models, selected actions, and paired macro transitions agreed. Maximum absolute Q difference was 3.0517578e-05 within the declared tolerance. TD-independent deployment was 5.257x faster overall (3.031x, 5.411x, and 5.572x by workload); certification alone was 6.131x faster. Macro execution and the training-only identity-deduplicated scoring and cached batched-TD accelerations were excluded from the deployment total. The sealed panel remained unopened. |

The final 8x8 path declares these accelerations and each must be traced to the
actual executed implementation before designing the benchmark:

1. relocation-family certification;
2. timing-invariant certificate caching and path cleanup;
3. device graph-input caching;
4. identity-deduplicated batched candidate scoring;
5. cached batched TD backup (training only).

The deployment runtime claim must exclude the TD-backup speed-up; the training
runtime report may include it separately.  Old 5x5 witness-reuse figures are
historical and cannot stand in for this test.

## Decisions requiring no new run

- VCG-PD is a writing decision based on existing results; it requires no new run.
- Disturbance robustness is dropped from the current claims and stated as a
  limitation.
- Transfer across sizes is dropped from the current claims and stated as future
  work. E195 and E215/E218 are independently trained scale-matched models.

## Remaining required work beyond currently partial jobs

1. Execute prepared E243: expanded matched-ranker 5x5 prefilled certification
   curve with 25 witnessed-safe days per fill level (400 frozen episodes).
2. Import and authenticate the E221/E223 artifacts, then execute the prepared
   E241 seed-1 first-divergence prefixes and report.

## Partial jobs currently running in user terminals

- E233: 143/230 new rows plus 120 referenced rows; certified Dynamic has
  23/30 panel rows, after which 80 stress rows remain.
- E235: 192/270 newly required rows; lambda 20 and 40 are complete, and
  lambda 160 seed 0 is at 12/30.
These commands are resumable by coordinate if interrupted. Their user-terminal
processes are not visible from the sandboxed process table, so running status
must be inferred from live log/artifact growth and terminal completion output,
not from `pgrep` alone.

## Recently completed

- E243 preparation: complete. The live four-route gate passed, all 100
  predeclared starts have exact nonempty completion witnesses, and focused
  tests pass. Remote execution of the 400 rows remains pending.

- E239: complete; 360/360 newly required lambda rows plus 90 reused lambda-10
  rows. Strict completion is 1.0, and the simple dial saturates at lambda 80.
- E240: complete; all equivalence checks passed across 36 decisions and 2,780
  candidate evaluations. TD-independent deployment was 5.257x faster overall,
  with 3.031x, 5.411x, and 5.572x speed-ups at 20%, 50%, and 80% workload.
  The sealed panel remained unopened.
- E236: seed-0 branch collection and head fitting are complete. Its failed
  head-only ranking screen is retained as a failed internal diagnostic.
- E242: complete; all 15 episodes completed strictly. Learned Stage B increased
  rehandles relative to both Stage A and immediate-only in every one of the six
  matched lambda/workload comparisons, so the counterfactual head is rejected.
