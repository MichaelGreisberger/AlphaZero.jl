Network = NetLib.ResNet

# netparams = NetLib.SimpleNetHP(
#   width=45,
#   depth_common=6,
#   use_batch_norm=true,
#   batch_norm_momentum=1.)

netparams = NetLib.ResNetHP(
  num_filters=128,
  num_blocks=5,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)


self_play = SelfPlayParams(
  sim=SimParams(
    num_games=5000,
    num_workers=128,
    use_gpu=true,
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=600,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=100,
    num_workers=100,
    use_gpu=true,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=false),
  mcts = MctsParams(
    self_play.mcts,
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.05)

learning = LearningParams(
  use_gpu=true,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=Adam(lr=2e-3),
  batch_size=1024,
  loss_computation_batch_size=1024,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=10,
  memory_analysis=MemAnalysisParams(
    num_game_stages=3),
  ternary_rewards=true,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(
  [      0,        15],
  [400_000, 1_000_000]))

#####
##### Evaluation benchmark
#####

mcts_baseline =
  Benchmark.MctsRollouts(
    MctsParams(
      arena.mcts,
      num_iters_per_turn=1000,
      cpuct=1.))

minmax_baseline = Benchmark.MinMaxTS(
  depth=5,
  τ=0.2,
  amplify_rewards=true)
  
alphazero_player = Benchmark.Full(arena.mcts)

network_player = Benchmark.NetworkOnly(τ=0.5)

benchmark_sim = SimParams(
  arena.sim;
  num_games=128,
  num_workers=128)

benchmark = [
    Benchmark.Duel(alphazero_player, mcts_baseline,   benchmark_sim),
    Benchmark.Duel(alphazero_player, minmax_baseline, benchmark_sim),
    Benchmark.Duel(network_player,   mcts_baseline,   benchmark_sim),
    Benchmark.Duel(network_player,   minmax_baseline, benchmark_sim)
]

experiment = Experiment(
  "fanorona", GameSpec(), params, Network, netparams, benchmark) 