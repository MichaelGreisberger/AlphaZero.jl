import AlphaZero.Scripts.dummy_run
import AlphaZero.Scripts.test_game
import AlphaZero.Scripts.scripts
include("game.jl")
# import AlphaZero
# include("../../src/scripts/dummy_run.jl")
# spec, env = run_test()
# println("TEST GAME")
# test_game(GameSpec())
println("DUMMY RUN")
dummy_run("fanorona")
# dummy_run("connect-four")
# println("Training")
# train("fanorona")