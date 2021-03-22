import AlphaZero.GI

using StaticArrays

const NUM_COLS = 0x09
const NUM_ROWS = 0x05
const NUM_CELLS = NUM_COLS * NUM_ROWS
const PIECES_PER_PLAYER = UInt8(floor(NUM_CELLS/2))

const NUM_ACTIONS = 1080

const Player = UInt8
const WHITE = 0x01
const BLACK = 0x02

other(p::Player) = 0x03 - p

@enum MoveType paika=0 approach=1 withdrawal=2

const Cell = UInt8
const EMPTY = 0x00
const Board = Vector{UInt8}

# const INITIAL_BOARD = SMatrix{9,5,UInt8,0x45}(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,0,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
const INITIAL_BOARD = [
    BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,WHITE,BLACK,WHITE,EMPTY,BLACK,WHITE,BLACK,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE
]

const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

#######################################
### IMPLEMENTATION OF GAMEINTERFACE ###
#######################################

struct GameSpec <: GI.AbstractGameSpec 
    # @enum BoardSize small=(3, 3) medium=(5, 5) large=(9, 5)
    possible_actions :: Array{UInt16}
    GameSpec() = new(calc_possible_actions())
end

mutable struct GameEnv <: GI.AbstractGameEnv
    board :: Board
    curplayer :: Player
    white_pieces :: UInt8
    black_pieces :: UInt8
    finished :: Bool
    winner :: Player
    action_mask :: Vector{Bool} # actions mask
    # Actions history, which uniquely identifies the current board position
    # Used by external solvers
end

function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  winner = 0x00
  action_mask = falses(1080)
  return GameEnv(board, curplayer, PIECES_PER_PLAYER, PIECES_PER_PLAYER, finished, winner, action_mask)
end

GI.spec(::GameEnv) = GameSpec()

#####
##### Queries on specs
#####
  
GI.two_players(::GameSpec) = true

GI.actions(spec::GameSpec) = spec.possible_actions

flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

function flip_colors(board::Board)
    return UInt8[
        flip_cell_color(board[position])
        for position in 1:NUM_CELLS
    ]
end

function GI.vectorize_state(::GameSpec, state::GameEnv)
    board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
    return Float32[
        board[position]
        for position in 1:NUM_CELLS
    ]
end

#####
##### Operations on envs
#####

# TODO: Update with functions accordingly!
function GI.set_state!(g::GameEnv, state)
    g.board = state.board
    g.curplayer = state.curplayer
    g.action_mask = calc_actions_mask(g)
end

function GI.current_state(g::GameEnv)
    return (board=copy(g.board), curplayer=g.curplayer)
end

function game_terminated(g::GameEnv)
    found_white = false;
    found_black = false;
    for position in 1:NUM_CELLS
        if g.board[position] == WHITE
            found_white = true
        end
        if g.board[position] == BLACK
            found_black = true
        end
    end
    return !(found_white && found_black)
end

function white_playing(g::GameEnv)
    return g.curplayer == WHITE
end

function action_mask(g::GameEnv)
    return g.action_mask
end

function play!(g::GameEnv, action)
    ### TODO: Implement
end

function white_reward(g::GameEnv)
    return g.white_pieces - g.black_pieces
end

function heuristic_value(g::GameEnv)
    if g.curplayer == WHITE
        return white_reward(g)
    else
        return -white_reward(g)
    end
end


#####
##### Interface for interactive exploratory tools
#####

function render(g::GameEnv)
    board = g.board
    buffer = IOBuffer()

    xLength = NUM_COLS * 0x02 - 0x01
    yLength = NUM_ROWS * 0x02 - 0x01

    println(buffer, "  a   b   c   d   e   f   g   h   i")
    for y in UnitRange(0x01, yLength)
        print(buffer, y % 2 != 0 ? string(floor(Int, y / 2)) * " " : "  ")
        for x in UnitRange(0x01, xLength)
            if x % 2 != 0
                if y % 2 != 0
                    pos = cord_to_pos(floor(UInt8, x/2 + 1), floor(UInt8, y/2 + 1))
                    val = board[pos]
                    print(buffer, val == 0 ? " " : val == 1 ? "W" : "B")
                else
                    print(buffer, "|")
                end
            else
                if y % 2 != 0
                    print(buffer, " - ")
                else
                    print(buffer, (x + y) % 4 == 2 ? " / " : " \\ ")
                end
            end
        end
        println(buffer, "")
    end
    return String(take!(buffer))
end

function action_string(::GameSpec, action::UInt16)
    
    # return (x0 << 12) + (y0 << 9) + (x1 << 5) + (y1 << 2) + UInt8(type)

    
end

######################
### HELPER METHODS ###
######################

function calc_possible_actions()    
    actions = Array{UInt16}(undef, NUM_ACTIONS)
    for position = UnitRange(0x01, NUM_CELLS), direction = UnitRange(0x01, 0x08)
        index = action_index(position, direction)
        actions[index] = action_value(position, direction, paika)
        actions[index + Int(approach)] = action_value(position, direction, approach)
        actions[index + Int(withdrawal)] = action_value(position, direction, withdrawal)
    end
    return actions
end

function action_index(position::UInt8, direction::UInt8)
    # 24 possible moves per position, 3 possible moves per direction
    return (position - 1) * 24 + 1 + (direction - 1) * 3
end 

function action_index(x::UInt8, y::UInt8, direction::UInt8)
    return action_index(cord_to_pos(x, y), direction)
end

function action_value(position::UInt8, direction::UInt8, type::MoveType)
    x0, y0 = pos_to_cord(position)
    x1 = UInt8
    y1 = UInt8
    (x1, y1) = new_coords(x0, y0, direction)
    return action_value(x0, y0, x1, y1, type)
end

function action_value(x0::UInt8, y0::UInt8, x1::UInt8, y1::UInt8, type::MoveType)    
    # println("from (x, y): ($x0, $y0), to: ($x1, $y1), type: $type, position: $position, direction: $direction")
    
    # x0 uses bit 16 to 13
    # y0 uses bit 12 to 10
    # x1 uses bit 9 to 6
    # y1 uses bit 5 to 3
    # type uses bit 2 to 1
    return (UInt16(x0) << 12) + (UInt16(y0) << 9) + (UInt16(x1) << 5) + (y1 << 2) + UInt8(type)
end

function decode_action_value(action::UInt16)
    type = MoveType(action & 0x03)
    y1 = (action >> 2) & 0x07
    x1 = (action >> 5) & 0x0f
    y0 = (action >> 9) & 0x07
    x0 = (action >> 12) & 0x0f
    return x0, y0, x1, y1, type
end

function new_coords(x0::UInt8, y0::UInt8, direction::UInt8)
    x1 = UInt8 # destination coordinate x (columns)
    y1 = UInt8 # destination coordinate y (rows)

    if direction == 1
        x1 = x0
        y1 = y0 - 0x01
    elseif direction == 2
        x1 = x0 + 0x01
        y1 = y0 - 0x01
    elseif direction == 3
        x1 = x0 + 0x01
        y1 = y0
    elseif direction == 4
        x1 = x0 + 0x01
        y1 = y0 + 0x01
    elseif direction == 5
        x1 = x0
        y1 = y0 + 0x01
    elseif direction == 6
        x1 = x0 - 0x01
        y1 = y0 + 0x01
    elseif direction == 7
        x1 = x0 - 0x01
        y1 = y0
    else
        x1 = x0 - 0x01
        y1 = y0 - 0x01
    end

    return (x1, y1)
end

function pos_to_cord(position::UInt8)
    x = (position - 0x01) % NUM_COLS + 0x01 # origin coordinate x (columns)
    y = UInt8(ceil(position / NUM_COLS)) # origin coordinate y (rows)
    return x, y
end

function cord_to_pos(x::UInt8, y::UInt8)
    return UInt8((y - 0x01) * NUM_COLS + x)
end

# Ich glaub das muss ich garnicht implementieren. wird der sanity check zeigen :)
# function GI.clone(g::GameEnv)
#     GameEnv(g.board, g.curplayer, g.finished, g.winner, copy(g.action_mask))
# end

function calc_actions_mask(g::GameEnv)
    g.action_mask = falses(NUM_ACTIONS)
    found_capture = false;
    paika_moves = Vector{Tuple}()

    player = g.curplayer
    opponent = other(player)

    for x = UnitRange(0x01, NUM_COLS), y = UnitRange(0x01, NUM_ROWS)
        if g.board[cord_to_pos(x, y)] == player
            for (x1, y1, d) in get_empty_neighbours(x, y, g.board)  
                if opponent_in_direction(x1, y1, d, opponent, g.board)
                    found_capture = true;
                    g.action_mask[action_index(x, y, d) + Int(approach)] = true
                    println(x, y, x1, y1, "approach")
                end
                if opponent_in_direction(x, y, UInt8((d + 4) % 8), opponent, g.board) 
                    #(d + 4) % 8 Is die opposit direction of d; we start from x, y because 
                    #opponent_in_direction checks whether an opponent piece is on the position 
                    #one hopp away in the specified direction. For withdrawal this is against the 
                    #direction of movement one hopp from the origin.
                    found_capture = true
                    g.action_mask[action_index(x, y, d) + UInt8(withdrawal)] = true
                    println(x, y, x1, y1, "withdrawal")
                end
                if !found_capture
                    print("paika")
                    push!(paika_moves, (x, y, d))
                end
            end
        end
    end

    if !found_capture
        for (x, y, d) in paika_moves
            g.action_mask[action_index(x, y, d) + Int(paika)] = true
        end
    end
end

function get_empty_neighbours(x::UInt8, y::UInt8, board::Board)
    empty_neighbours = Vector{Tuple}()
    for (x1, y1, d) in get_surrounding_nodes(x, y) 
        if board[cord_to_pos(x1, y1)] == EMPTY
            push!(empty_neighbours, (x1, y1, d))
        end
    end
    return empty_neighbours
end

function get_surrounding_nodes(x::UInt8, y::UInt8)
    if (is_strong_node(x, y))
        return get_eight_neighbours(x, y)
    else
        return get_four_neighbours(x, y)
    end
end

function is_strong_node(x::UInt8, y::UInt8)
    return (x + y * NUM_COLS) % 2 == 0
end

function get_eight_neighbours(x::UInt8, y::UInt8)
    neighbours = Tuple[]
    if (is_in_board_space(x, y - 0x01))
        push!(neighbours, (x, y - 0x01, 0x01))
    end
    if (is_in_board_space(x + 0x01, y - 0x01))
        push!(neighbours, (x + 0x01, y - 0x01, 0x02))
    end
    if (is_in_board_space(x + 0x01, y))
        push!(neighbours, (x + 0x01, y, 0x03))
    end
    if (is_in_board_space(x + 0x01, y + 0x01))
        push!(neighbours, (x + 0x01, y + 0x01, 0x04))
    end
    if (is_in_board_space(x, y + 0x01))
        push!(neighbours, (x, y + 0x01, 0x05))
    end
    if (is_in_board_space(x - 0x01, y + 0x01))
        push!(neighbours, (x - 0x01, y + 0x01, 0x06))
    end
    if (is_in_board_space(x - 0x01, y))
        push!(neighbours, (x - 0x01, y, 0x07))
    end
    if (is_in_board_space(x - 0x01, y - 0x01))
        push!(neighbours, (x - 0x01, y - 0x01, 0x08))
    end
    return neighbours
end

function get_four_neighbours(x::UInt8, y::UInt8)
    neighbours = Tuple[]
    if (is_in_board_space(x, y - 0x01))
        push!(neighbours, (x, y - 0x01, 0x01))
    end
    if (is_in_board_space(x + 0x01, y))
        push!(neighbours, (x + 0x01, y, 0x03))
    end
    if (is_in_board_space(x, y + 0x01))
        push!(neighbours, (x, y + 0x01, 0x05))
    end
    if (is_in_board_space(x - 0x01, y))
        push!(neighbours, (x - 0x01, y, 0x07))
    end
    return neighbours
end

function is_in_board_space(x::UInt8, y::UInt8)    
    return x > 0 && x <= NUM_COLS && y > 0 && y < NUM_ROWS
end

function opponent_in_direction(x::UInt8, y::UInt8, d::UInt8, opponent::UInt8, board::Board)
    x1, y1 = new_coords(x, y, d)
    return board[cord_to_pos(x1, y1)] == opponent
end



####################
### TEST METHODS ###
####################

function run_test()
    spec = GameSpec()
    env = GI.init(spec)
    calc_actions_mask(env)
    print(env.action_mask)
    return spec, env
end

spec, env = run_test()