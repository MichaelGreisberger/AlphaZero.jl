import AlphaZero.GI

using StaticArrays

const NUM_COLS = 0x0009
const NUM_ROWS = 0x0005
const NUM_CELLS = NUM_COLS * NUM_ROWS
const PIECES_PER_PLAYER = UInt16(floor(NUM_CELLS/2))

const NUM_ACTIONS = NUM_CELLS * 8 * 3 + 1

const Player = UInt16
const WHITE = 0x0001
const BLACK = 0x0002

other(p::Player) = 0x0003 - p

@enum ActionType  paika=0 approach=1 withdrawal=2 pass=3

const Cell = UInt16
const EMPTY = 0x00
const Board = Vector{UInt16}

# const INITIAL_BOARD = SMatrix{9,5,UInt16,0x45}(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,0,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
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
    white_pieces :: UInt16
    black_pieces :: UInt16
    finished :: Bool
    winner :: Player
    action_mask :: Vector{Bool}
    extended_capture :: Bool
    last_positions :: Array{Tuple}
    last_direction :: UInt8
end

function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  winner = 0x00
  env = GameEnv(board, curplayer, PIECES_PER_PLAYER, PIECES_PER_PLAYER, finished, winner, [], false, [], 0x00)
  update_actions_mask!(env)
  return env
end

GI.spec(::GameEnv) = GameSpec()

#####
##### Queries on specs
#####
  
GI.two_players(::GameSpec) = true

GI.actions(spec::GameSpec) = spec.possible_actions

flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

function flip_colors(board::Board)
    return UInt16[
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
    g.action_mask = update_actions_mask!(g)
end

function GI.current_state(g::GameEnv)
    return (board=copy(g.board), curplayer=g.curplayer)
end

#can be simplified with pieces left
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

function GI.play!(g::GameEnv, action::UInt16)
    ### TODO: Implement
    x0, y0, x1, y1, type = decode_action_value(action)
    if type == pass
        swap_player(g)
    elseif type == paika
        apply_action(g, x0, y0, x1, y1, type)
    else
        apply_action(g, x0, y0, x1, y1, type)

        g.extended_capture = true
        push!(g.last_positions, (x0, y0))
        g.last_direction = calc_direction(x0, y0, x1, y1)
        
        found_capture = update_actions_mask!(g)
        
        if !found_capture
            swap_player(g)
        end
    end
end

function apply_action(g::GameEnv, x0::UInt16, y0::UInt16, x1::UInt16, y1::UInt16, type::ActionType)
    if type == approach
        dirX = Int(x1) - x0
        dirY = Int(y1) - y0
        capture(g, x1, y1, dirX, dirY, other(g.curplayer))
    elseif type == withdrawal
        dirX = Int(x0) - x1
        dirY = Int(y0) - y1
        capture(g, x1 + 0x02 * dirX, y1 + 0x02 * dirY, dirX, dirY, other(g.curplayer))
    end
    g.board[cord_to_pos(x0, y0)] = 0x0000
    g.board[cord_to_pos(x1, y1)] = g.curplayer
end

function swap_player(g::GameEnv)
    g.curplayer = other(g.curplayer)
    g.extended_capture = false
    g.last_direction = 0x00
    g.last_positions = []
    update_actions_mask!(g)
end

function capture(g::GameEnv, x, y, dirX, dirY, opponent::UInt16)
    g.board[cord_to_pos(x, y)] = 0x0000
    if opponent == WHITE
        g.white_pieces -= 1
    else
        g.black_pieces -= 1
    end

    x1 = x + dirX
    y1 = y + dirY

    if is_in_board_space(x1, y1) && g.board[cord_to_pos(x1, y1)] == opponent
        capture(g, x1, y1, dirX, dirY, opponent)
    end
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
    for y in UnitRange(0x0001, yLength)
        print(buffer, y % 2 != 0 ? string(ceil(Int, y / 2)) * " " : "  ")
        for x in UnitRange(0x0001, xLength)
            if x % 2 != 0
                if y % 2 != 0
                    pos = cord_to_pos(floor(UInt16, x/2 + 1), floor(UInt16, y/2 + 1))
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
    x0, y0, x1, y1, type = decode_action_value(action)
    if type == pass
        return "pass"
    else
        return Char(x0 + 96) * string(y0) * Char(x1 + 96) * string(y1) * (type == paika ? 'P' : type == approach ? 'A' : type == withdrawal ? 'W' : error())
    end
end

function read_state(::GameSpec)
    board = Board(undef, NUM_CELLS)
    try
        i = 1
        while true
            if i > NUM_CELLS
                print("done")
                return board
            end
            readline() #row not important
            input = readline()
            for (col, val) in enumerate(input)
                if (col % 4 == 3)
                    val = lowercase(val)
                    println("val $val col $col index $i")
                    if val == 'w'
                        board[i] = WHITE
                    elseif val == 'b'
                        board[i] = BLACK
                    end
                    i += 1
                end
            end
        end
        #TODO: find way to know whos the current player..
        #maybe the only way is to encode in input..
    catch e
        return nothing
    end
end

######################
### HELPER METHODS ###
######################

function calc_possible_actions()    
    actions = Array{UInt16}(undef, NUM_ACTIONS)
    for position = UnitRange(0x0001, NUM_CELLS), direction = UnitRange(0x01, 0x08)
        index = action_index(position, direction)
        actions[index + Int(paika)] = action_value(position, direction, paika)
        actions[index + Int(approach)] = action_value(position, direction, approach)
        actions[index + Int(withdrawal)] = action_value(position, direction, withdrawal)
    end
    return actions
end

function action_index(position::UInt16, direction::UInt8)
    # 24 possible moves per position, 3 possible moves per direction
    return (position - 1) * 24 + 1 + (direction - 1) * 3
end 

function action_index(x::UInt16, y::UInt16, direction::UInt8)
    return action_index(cord_to_pos(x, y), direction)
end

function action_value(position::UInt16, direction::UInt8, type::ActionType)
    x0, y0 = pos_to_cord(position)
    x1 = UInt16
    y1 = UInt16
    (x1, y1) = new_coords(x0, y0, direction)
    return action_value(x0, y0, x1, y1, type)
end

function action_value(x0::UInt16, y0::UInt16, x1::UInt16, y1::UInt16, type::ActionType)    
    # println("from (x, y): ($x0, $y0), to: ($x1, $y1), type: $type, position: $position, direction: $direction")
    
    # x0 uses bit 16 to 13
    # y0 uses bit 12 to 10
    # x1 uses bit 9 to 6
    # y1 uses bit 5 to 3
    # type uses bit 2 to 1
    return (UInt16(x0) << 12) + (UInt16(y0) << 9) + (UInt16(x1) << 5) + (y1 << 2) + UInt16(type)
end

function decode_action_value(action::UInt16)
    if action == 0x0000
        return 0, 0, 0, 0, pass
    end
    type = ActionType(action & 0x03)
    y1 = (action >> 2) & 0x07
    x1 = (action >> 5) & 0x0f
    y0 = (action >> 9) & 0x07
    x0 = (action >> 12) & 0x0f
    return x0, y0, x1, y1, type
end

function new_coords(x0::UInt16, y0::UInt16, direction::UInt8)
    x1 = UInt16 # destination coordinate x (columns)
    y1 = UInt16 # destination coordinate y (rows)

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

function calc_direction(x0::UInt16, y0::UInt16, x1::UInt16, y1::UInt16)
    if x0 < x1 #+x
        if y0 < y1 #+y
            return 0x04
        elseif y0 == y1
            return 0x03
        else
            return 0x02
        end
    elseif x1 < x0 #-x
        if y0 < y1 #+y
            return 0x06
        elseif y0 == y1
            return 0x07
        else
            return 0x08
        end
    else        
        if y0 < y1 #+y
            return 0x05
        else
            return 0x01
        end
    end
end

function pos_to_cord(position::UInt16)
    x = (position - 0x01) % NUM_COLS + 0x01 # origin coordinate x (columns)
    y = UInt16(ceil(position / NUM_COLS)) # origin coordinate y (rows)
    return x, y
end

function cord_to_pos(x::UInt16, y::UInt16)
    return UInt16((y - 0x01) * NUM_COLS + x)
end

# Ich glaub das muss ich garnicht implementieren. wird der sanity check zeigen :)
# function GI.clone(g::GameEnv)
#     GameEnv(g.board, g.curplayer, g.finished, g.winner, copy(g.action_mask))
# end

function update_actions_mask!(g::GameEnv)
    g.action_mask = falses(NUM_ACTIONS)
    found_capture = false;
    paika_moves = Vector{Tuple}()

    player = g.curplayer
    opponent = other(player)
    
    if g.extended_capture
        x, y = last(g.last_positions)
        for (x1, y1, d) in get_empty_neighbours(x, y, g.board)
            if d != g.last_direction & !((x, y) ∈ g.last_positions)
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
                    g.action_mask[action_index(x, y, d) + UInt16(withdrawal)] = true
                    println(x, y, x1, y1, "withdrawal")
                end
                if !found_capture
                    print("paika")
                    push!(paika_moves, (x, y, d))
                end
            end
        end
    else
        for x = UnitRange(0x0001, NUM_COLS), y = UnitRange(0x0001, NUM_ROWS)
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
                        g.action_mask[action_index(x, y, d) + UInt16(withdrawal)] = true
                        println(x, y, x1, y1, "withdrawal")
                    end
                    if !found_capture
                        print("paika")
                        push!(paika_moves, (x, y, d))
                    end
                end
            end
        end
    end

    if !found_capture
        for (x, y, d) in paika_moves
            g.action_mask[action_index(x, y, d) + Int(paika)] = true
        end
    end

    if g.extended_capture
        g.action_mask[NUM_ACTIONS] = true #pass
    end

    return found_capture
end

function get_empty_neighbours(x::UInt16, y::UInt16, board::Board)
    empty_neighbours = Vector{Tuple}()
    for (x1, y1, d) in get_surrounding_nodes(x, y) 
        if board[cord_to_pos(x1, y1)] == EMPTY
            push!(empty_neighbours, (x1, y1, d))
        end
    end
    return empty_neighbours
end

function get_surrounding_nodes(x::UInt16, y::UInt16)
    if (is_strong_node(x, y))
        return get_eight_neighbours(x, y)
    else
        return get_four_neighbours(x, y)
    end
end

function is_strong_node(x::UInt16, y::UInt16)
    return (x + y * NUM_COLS) % 2 == 0
end

function get_eight_neighbours(x::UInt16, y::UInt16)
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

function get_four_neighbours(x::UInt16, y::UInt16)
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

function is_in_board_space(x::UInt16, y::UInt16)    
    return x > 0 && x <= NUM_COLS && y > 0 && y < NUM_ROWS
end

function opponent_in_direction(x::UInt16, y::UInt16, d::UInt8, opponent::UInt16, board::Board)
    x1, y1 = new_coords(x, y, d)
    
    if is_in_board_space(x1, y1)
        return board[cord_to_pos(x1, y1)] == opponent
    else
        return false
    end
end

function print_possible_action_strings(spec::GameSpec, g::GameEnv)
    for action in GI.actions(spec)[g.action_mask]
        action_hex = "0x" * string(action, base=16)
        println(action_string(spec, action) * " ($action_hex)")
    end
end

function print_possible_action(spec::GameSpec, g::GameEnv)
    GI.actions(spec)[g.action_mask]
end

####################
### TEST METHODS ###
####################

function run_test()
    spec = GameSpec()
    env = GI.init(spec)
    while !game_terminated(env)
        println(render(env))
        println("its player " * (env.curplayer == WHITE ? " white's " : " black's ") * "turn!")
        println("possible moves are: ")
        print_possible_action_strings(spec, env)
        input = parse(UInt16, readline())
        GI.play!(env, input)
    end
end

run_test()