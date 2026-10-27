E139 SEQUENTIAL OVERNIGHT REPLICATES OF FROZEN E138
=================================================
User-authorized speculative replication: spend the compute now; retain failures
and all seeds, whether or not the recipe works. No automatic erasure, best-seed
selection or recipe tuning. No long training was launched while preparing this.

    cd /home/ai_diagnosis/HRL_Management
    tmux new -s e139-night
    bash experiments/conditioned_vcg/E139_grid_mixture_three_seeds_203k_v1/night.sh

Queue: wait for an already active seed-0 E138 training process, then finish/resume
seed 0 through episode 60, then train seed 1 through 60, then seed 2 through 60.
All greedy validations are included. Each seed starts from scratch unless its
own resumable checkpoint exists. No running E138 source/contract is changed.
If seed 0 is already complete, its command performs no new training.

The wait recognizes the standard E138 run.sh Python command; it checks every
60 seconds, stops after 24h waiting, and does not launch another writer while
that command remains active. Use this queue once; do not launch an additional
manual E138/E139 trainer while it runs. flock prevents duplicate night queues.
Detach tmux with Ctrl-B then D. Attach with: tmux attach -t e139-night

Each seed retains E138's stop-on-timeout/failure/nonfinite rule. A failed seed
is NOT retried automatically: its results remain, then the next independent
seed may start, as explicitly authorized for overnight compute utilization.
Ctrl-C/SIGTERM stops the QUEUE, not merely the current seed. Final queue exit
is nonzero if any child failed. Three planned seeds need not mean three completed
seeds; inspect stopped.json and completion/validation counts before conclusions.

Seed 0 remains at the E138 -repair1 output. Additional seeds are stored at:
    results/vcg-conditioned-e139-grid-mixture-three-seeds-203k-v1/seed-1/
    results/vcg-conditioned-e139-grid-mixture-three-seeds-203k-v1/seed-2/

    bash experiments/conditioned_vcg/E139_grid_mixture_three_seeds_203k_v1/run.sh status

Or launch one additional seed manually (not concurrently with the queue):
    bash experiments/conditioned_vcg/E139_grid_mixture_three_seeds_203k_v1/run.sh train --model-seed 1 --through-episode 60 --device cuda

Same frozen recipe, no new controller changes: .90->.05 over 6000 training
decisions, 60 balanced-regime episodes, every-six greedy development validation,
same workload seeds/regime ordering and shared dev panel. Different MODEL seeds
change initialization, exploration, replay sampling and global RNG. These are
independent optimization replicates with matched exogenous workloads, NOT extra
independent evaluation instances. Source contracts and model-seed checks prevent
cross-loading checkpoints. Truncated episode updates roll back to the last
completed save, like E138. No test panel, Stage B or deployment/paper eligibility.

Runtime: sequential, not simultaneous on the same GPU. Three full pilots can
take most of a night or longer; this is not a promise they finish by morning.
Any changes made after reviewing their results require a separately versioned
recipe, and these original outcomes should still be retained/reported.
