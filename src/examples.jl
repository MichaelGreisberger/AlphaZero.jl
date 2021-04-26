module Examples

  using ..AlphaZero

  include("../games/tictactoe/main.jl")
  export Tictactoe

  include("../games/connect-four/main.jl")
  export ConnectFour

  include("../games/grid-world/main.jl")
  export GridWorld

  include("../games/fanorona/main.jl")
  export Fanorona

  const games = Dict(
    "grid-world" => GridWorld.GameSpec(),
    "tictactoe" => Tictactoe.GameSpec(),
    "fanorona" => Fanorona.GameSpec(),
    "connect-four" => ConnectFour.GameSpec())

  const experiments = Dict(
    "grid-world" => GridWorld.Training.experiment,
    "tictactoe" => Tictactoe.Training.experiment,
    "fanorona" => Fanorona.Training.experiment,
    "connect-four" => ConnectFour.Training.experiment)

end