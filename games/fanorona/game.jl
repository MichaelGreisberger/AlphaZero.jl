import AlphaZero.GI
using StaticArrays

@enum ActionType paika=0 approach=1 withdrawal=2 pass=3
@enum BoardSize small medium large

const size = large
const NUM_COLS = size == small ? 3 : size == medium ? 5 : 9
const NUM_ROWS = size == small ? 3 : 5

const NUM_CELLS = NUM_COLS * NUM_ROWS
const PIECES_PER_PLAYER = Int(floor(NUM_CELLS/2))

const NUM_ACTIONS = NUM_CELLS * 8 * 3 + 1

const Player = UInt8
const WHITE = 1
const BLACK = 2

other(p::Player) = 0x03 - p

const Cell = UInt8
const Action = UInt16
const Coord = UInt8
const EMPTY = 0

const Board = SVector{NUM_CELLS, Cell}

const INITIAL_BOARD_9x5 = SVector{45, Cell}(
    BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,WHITE,BLACK,WHITE,EMPTY,BLACK,WHITE,BLACK,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE
)

const INITIAL_BOARD_5x5 = SVector{25, Cell}(
    BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,WHITE,EMPTY,BLACK,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE
)

const INITIAL_BOARD_3x3 = SVector{9, Cell}(
    BLACK,BLACK,BLACK,
    BLACK,EMPTY,WHITE,
    WHITE,WHITE,WHITE
)

const INITIAL_STATE = (board= size == small ? INITIAL_BOARD_3x3 : size == medium ? INITIAL_BOARD_5x5 : INITIAL_BOARD_9x5, current_player=WHITE)

#######################################
### IMPLEMENTATION OF GAMEINTERFACE ###
#######################################

struct GameSpec <: GI.AbstractGameSpec 
    possible_actions :: Array{Action}
    GameSpec() = new(calc_possible_actions())
end

mutable struct GameEnv <: GI.AbstractGameEnv
    spec :: GameSpec
    board :: Board
    current_player :: Player
    white_pieces :: Int
    black_pieces :: Int
    action_mask :: Array{Bool}
    extended_capture :: Bool
    current_position :: Tuple
    last_positions :: Array{Tuple}
    last_direction :: Int
end

function GI.init(spec::GameSpec)
  board = copy(INITIAL_STATE.board)
  current_player = INITIAL_STATE.current_player
  env = GameEnv(spec, board, current_player, PIECES_PER_PLAYER, PIECES_PER_PLAYER, [], false, (0, 0), [], 0)
  update_actions_mask!(env)
  return env
end

GI.spec(g::GameEnv) = g.spec

#####
##### Queries on specs
#####
  
GI.two_players(::GameSpec) = true

GI.actions(spec::GameSpec) = spec.possible_actions

function GI.vectorize_state(::GameSpec, state)
    board = state.current_player == WHITE ? state.board : flip_colors(state.board)
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
    g.current_player = state.current_player
    g.last_direction = state.last_direction
    g.last_positions = copy(state.last_positions) #this copy is necessary
    g.extended_capture = state.extended_capture
    g.current_position = (state.current_position[1], state.current_position[2])
    g.white_pieces = count(==(WHITE), g.board)
    g.black_pieces = count(==(BLACK), g.board)
    if g.extended_capture
        update_actions_mask_extended_capture!(g)
    else
        update_actions_mask!(g)
    end
end

function GI.current_state(g::GameEnv)
    state_copy = (board=copy(g.board), 
        current_player=g.current_player, 
        extended_capture=g.extended_capture, 
        last_positions=copy(g.last_positions), 
        last_direction=g.last_direction,
        current_position=(g.current_position[1], g.current_position[2])
    )
    return state_copy
end

#can be simplified with pieces left
function GI.game_terminated(g::GameEnv)
    return !(g.white_pieces > 0 && g.black_pieces > 0)
end

function GI.white_playing(g::GameEnv)
    return g.current_player == WHITE
end

function GI.actions_mask(g::GameEnv)
    # print_state_information(g)
    return g.action_mask
end

function GI.play!(g::GameEnv, action::Action)
    x0, y0, x1, y1, type = decode_action(action)
    if type == pass
        swap_player(g)
    elseif type == paika
        apply_action(g, x0, y0, x1, y1, type)
        swap_player(g)
    else
        apply_action(g, x0, y0, x1, y1, type)

        g.extended_capture = true       
        g.last_direction = determine_direction(x0, y0, x1, y1)
        push!(g.last_positions, (x0, y0))
        g.current_position = (x1, y1)

        found_capture = update_actions_mask_extended_capture!(g)
        
        if !found_capture
            swap_player(g)
        end
    end
    # println(GI.render(g))
    # print_possible_action_strings(g)
end

function GI.white_reward(g::GameEnv)
    if GI.game_terminated(g)
        return g.white_pieces > 0.0 ? 1.0 : -1.0
    else
        return 0
    end
end

function white_heuristic_value(g::GameEnv)
    return Float64(g.white_pieces - g.black_pieces)
end

function GI.heuristic_value(g::GameEnv)
    if g.current_player == WHITE
        return white_heuristic_value(g)
    else
        return -white_heuristic_value(g)
    end
end


#####
##### Interface for interactive exploratory tools
#####

function GI.render(g::GameEnv)
    board = g.board
    buffer = IOBuffer()

    xLength = NUM_COLS * 02 - 01
    yLength = NUM_ROWS * 02 - 01

    player = g.current_player == WHITE ? 'W' : 'B'

    if size == small
        println(buffer, player * " a   b   c")
    elseif size == medium
        println(buffer, player * " a   b   c   d   e")
    else
        println(buffer, player * " a   b   c   d   e   f   g   h   i")
    end

    for y in UnitRange(0001, yLength)
        print(buffer, y % 2 != 0 ? string(ceil(Int, y / 2)) * " " : "  ")
        for x in UnitRange(0001, xLength)
            if x % 2 != 0
                if y % 2 != 0
                    pos = cord_to_pos(floor(Int, x/2 + 1), floor(Int, y/2 + 1))
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

function GI.action_string(::GameSpec, action::Action)
    x0, y0, x1, y1, type = decode_action(action)
    if type == pass
        return "pass"
    else
        return Char(x0 + 96) * string(y0) * Char(x1 + 96) * string(y1) * (type == paika ? 'P' : type == approach ? 'A' : type == withdrawal ? 'W' : error())
    end
end

function GI.parse_action(::GameSpec, str::String)
    if str == "pass"
        return 0
    else
        parts = collect(Char, str)
        return encode_action(Int(parts[1] - 96), parse(Int, parts[2]), Int(parts[3] - 96), parse(Int, parts[4]), parts[5] == 'P' ? paika : parts[5] == 'A' ? approach : withdrawal)
    end
end

function GI.read_state(::GameSpec)
    board = Board(undef, NUM_CELLS)
    println(board)
    try
        i = 1
        input = lowercase(readline())
        player = input[1] == 'w' ? WHITE : input[1] == 'b' ? BLACK : error("First character must either be 'W' or 'B' to identify the current player")
        linecount = size == small ? 6 : 10
        line = 1

        while line < linecount
            input = lowercase(readline())
            print("read line: $input")
            line += 1
            if line % 2 == 0
                for (col, val) in enumerate(input)
                    if (col % 4 == 3)
                        # println("val $val col $col index $i")
                        if val == 'w'
                            board[i] = WHITE
                        elseif val == 'b'
                            board[i] = BLACK
                        else 
                            board[i] = EMPTY
                        end
                        i += 1
                    end
                end
            end
        end
        return (board, player)
    catch e
        println("Error while reading input: $e")
        return nothing
    end
end

######################
### HELPER METHODS ###
######################


function apply_action(g::GameEnv, x0::Int, y0::Int, x1::Int, y1::Int, type::ActionType)
    if type == approach
        dirX = x1 - x0
        dirY = y1 - y0
        capture(g, x1 + dirX, y1 + dirY, dirX, dirY, other(g.current_player))
    elseif type == withdrawal
        dirX = x0 - x1
        dirY = y0 - y1
        capture(g, x1 + 02 * dirX, y1 + 02 * dirY, dirX, dirY, other(g.current_player))
    end
    # g.board[cord_to_pos(x0, y0)] = 0
    # g.board[cord_to_pos(x1, y1)] = g.current_player
    g.board = setindex(g.board, EMPTY, cord_to_pos(x0, y0))
    g.board = setindex(g.board, g.current_player, cord_to_pos(x1, y1))
end

function swap_player(g::GameEnv)
    g.current_player = other(g.current_player)
    g.extended_capture = false
    # print("last positions before reset: $(g.last_positions), ")
    g.last_positions = []
    # println("last positions after reset: $(g.last_positions)")
    g.last_direction = 0
    g.current_position = (0, 0)
    update_actions_mask!(g)
end

function capture(g::GameEnv, x, y, dirX, dirY, opponent::Player)
    # g.board[cord_to_pos(x, y)] = 0
    g.board = setindex(g.board, 0, cord_to_pos(x, y))
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

flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

function flip_colors(board::Board)
    return Int[
        flip_cell_color(board[position])
        for position in 1:NUM_CELLS
    ]
end

function calc_possible_actions()    
    actions = Array{Action}(undef, NUM_ACTIONS)
    for position = UnitRange(1, NUM_CELLS), direction = UnitRange(1, 8)
        index = action_index(position, direction)
        actions[index + Int(paika)] = encode_action(position, direction, paika)
        actions[index + Int(approach)] = encode_action(position, direction, approach)
        actions[index + Int(withdrawal)] = encode_action(position, direction, withdrawal)
    end
    actions[NUM_ACTIONS] = 0
    return actions
end

function action_index(position::Int, direction::Int)
    # 24 possible moves per position, 3 possible moves per direction
    return (position - 1) * 24 + 1 + (direction - 1) * 3
end 

function action_index(x::Int, y::Int, direction::Int)
    return action_index(cord_to_pos(x, y), direction)
end

function encode_action(position::Int, direction::Int, type::ActionType)
    # println("position: $position, direction: $direction, tpye: $type")
    x0, y0 = pos_to_cord(position)
    x1 = Int
    y1 = Int
    (x1, y1) = new_coords(x0, y0, direction)
    return encode_action(x0, y0, x1, y1, type)
end

function encode_action(x0::Int, y0::Int, x1::Int, y1::Int, type::ActionType)    
    # println("from (x, y): ($x0, $y0), to: ($x1, $y1), type: $type")
    
    # x0 uses bit 16 to 13
    # y0 uses bit 12 to 10
    # x1 uses bit 9 to 6
    # y1 uses bit 5 to 3
    # type uses bit 2 to 1
    return convert(Action, (x0 << 12) + (y0 << 9) + (x1 << 5) + (y1 << 2) + UInt8(type))
end

function decode_action(action::Action)
    if action == 0x0000
        return 0, 0, 0, 0, pass
    end
    type = ActionType(action & 3)
    y1 = (action >> 2) & 7
    x1 = (action >> 5) & 15
    y0 = (action >> 9) & 7
    x0 = (action >> 12) & 15
    return x0, y0, x1, y1, type
end

function new_coords(x0::Int, y0::Int, direction::Int)
    x1 = Int # destination coordinate x (columns)
    y1 = Int # destination coordinate y (rows)

    if direction == 1
        x1 = x0
        y1 = y0 - 01
    elseif direction == 2
        x1 = x0 + 01
        y1 = y0 - 01
    elseif direction == 3
        x1 = x0 + 01
        y1 = y0
    elseif direction == 4
        x1 = x0 + 01
        y1 = y0 + 01
    elseif direction == 5
        x1 = x0
        y1 = y0 + 01
    elseif direction == 6
        x1 = x0 - 01
        y1 = y0 + 01
    elseif direction == 7
        x1 = x0 - 01
        y1 = y0
    else
        x1 = x0 - 01
        y1 = y0 - 01
    end

    return (x1, y1)
end

function determine_direction(x0::Int, y0::Int, x1::Int, y1::Int)
    if x0 < x1 #+x
        if y0 < y1 #+y
            return 04
        elseif y0 == y1
            return 03
        else
            return 02
        end
    elseif x1 < x0 #-x
        if y0 < y1 #+y
            return 06
        elseif y0 == y1
            return 07
        else
            return 08
        end
    else        
        if y0 < y1 #+y
            return 05
        else
            return 01
        end
    end
end

function pos_to_cord(position::Int)
    x = (position - 01) % NUM_COLS + 01 # origin coordinate x (columns)
    y = Int(ceil(position / NUM_COLS)) # origin coordinate y (rows)
    return x, y
end

function cord_to_pos(x::Int, y::Int)
    return Int((y - 01) * NUM_COLS + x)
end

function update_actions_mask!(g::GameEnv)
    g.action_mask = falses(NUM_ACTIONS)

    found_capture = false
    paika_moves = Vector{Tuple}()

    player = g.current_player
    opponent = other(player)
    
    for x = UnitRange(1, NUM_COLS), y = UnitRange(1, NUM_ROWS)
        if g.board[cord_to_pos(x, y)] == player
            for (x1, y1, d) in get_empty_neighbours(x, y, g.board)
                found_capture |= update_action_mask_for_neighbour!(g, x, y, x1, y1, d, opponent)
                if !found_capture
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
    
    return found_capture
end

function update_actions_mask_extended_capture!(g::GameEnv)
    g.action_mask = falses(NUM_ACTIONS)
    opponent = other(g.current_player)
    found_capture = false

    for (x1, y1, d) in get_empty_neighbours(g.current_position[1], g.current_position[2], g.board)
        if d != g.last_direction && !((x1, y1) ∈ g.last_positions)
            found_capture |= update_action_mask_for_neighbour!(g, g.current_position[1], g.current_position[2], x1, y1, d, opponent)
        end
    end

    g.action_mask[NUM_ACTIONS] = true #pass
    
    return found_capture
end

function update_action_mask_for_neighbour!(g::GameEnv, x0::Int, y0::Int, x1::Int, y1::Int, d::Int, opponent::Player)
    found_capture = false;
    if opponent_in_direction(x1, y1, d, opponent, g.board)
        found_capture = true;
        g.action_mask[action_index(x0, y0, d) + Int(approach)] = true
    end
    if opponent_in_direction(x0, y0, opposite_direction(d), opponent, g.board) 
        found_capture = true
        g.action_mask[action_index(x0, y0, d) + Int(withdrawal)] = true
    end
    return found_capture
end

function opposite_direction(d::Int)
    return (d + 4) % 8
end

function get_empty_neighbours(x::Int, y::Int, board::Board)
    empty_neighbours = Vector{Tuple}()
    for (x1, y1, d) in get_surrounding_nodes(x, y) 
        if board[cord_to_pos(x1, y1)] == EMPTY
            push!(empty_neighbours, (x1, y1, d))
        end
    end
    return empty_neighbours
end

function get_surrounding_nodes(x::Int, y::Int)
    if (is_strong_node(x, y))
        return get_eight_neighbours(x, y)
    else
        return get_four_neighbours(x, y)
    end
end

function is_strong_node(x::Int, y::Int)
    return (x + y * NUM_COLS) % 2 == 0
end

function get_eight_neighbours(x::Int, y::Int)
    neighbours = Tuple[]
    if (is_in_board_space(x, y - 01))
        push!(neighbours, (x, y - 01, 01))
    end
    if (is_in_board_space(x + 01, y - 01))
        push!(neighbours, (x + 01, y - 01, 02))
    end
    if (is_in_board_space(x + 01, y))
        push!(neighbours, (x + 01, y, 03))
    end
    if (is_in_board_space(x + 01, y + 01))
        push!(neighbours, (x + 01, y + 01, 04))
    end
    if (is_in_board_space(x, y + 01))
        push!(neighbours, (x, y + 01, 05))
    end
    if (is_in_board_space(x - 01, y + 01))
        push!(neighbours, (x - 01, y + 01, 06))
    end
    if (is_in_board_space(x - 01, y))
        push!(neighbours, (x - 01, y, 07))
    end
    if (is_in_board_space(x - 01, y - 01))
        push!(neighbours, (x - 01, y - 01, 08))
    end
    return neighbours
end

function get_four_neighbours(x::Int, y::Int)
    neighbours = Tuple[]
    if (is_in_board_space(x, y - 01))
        push!(neighbours, (x, y - 01, 01))
    end
    if (is_in_board_space(x + 01, y))
        push!(neighbours, (x + 01, y, 03))
    end
    if (is_in_board_space(x, y + 01))
        push!(neighbours, (x, y + 01, 05))
    end
    if (is_in_board_space(x - 01, y))
        push!(neighbours, (x - 01, y, 07))
    end
    return neighbours
end

function is_in_board_space(x::Int, y::Int)    
    return x > 0 && x <= NUM_COLS && y > 0 && y <= NUM_ROWS
end

function opponent_in_direction(x::Int, y::Int, d::Int, opponent::Player, board::Board)
    x1, y1 = new_coords(x, y, d)
    
    if is_in_board_space(x1, y1)
        return board[cord_to_pos(x1, y1)] == opponent
    else
        return false
    end
end

function print_state_information(g::GameEnv)
    println("state: $(GI.current_state(g))")
    print_possible_action_strings(g)
end

function print_possible_action_strings(g::GameEnv)
    print("possible actions: ")
    for action in GI.actions(g.spec)[g.action_mask]
        print(GI.action_string(g.spec, action) * " ($action), ")
    end
    println()
end

function print_possible_action(g::GameEnv)
    GI.actions(g.spec)[g.action_mask]
end

####################
### TEST METHODS ###
####################

function run_test()
    spec = GameSpec()
    env = GI.init(spec)
    while !GI.game_terminated(env)
        # println(GI.render(env))
        # println("its player $(env.current_player == WHITE ? " white's " : " black's ") turn!")
        # print_possible_action_strings(env)
        possible_actions = GI.actions(spec)[env.action_mask]
        GI.play!(env, rand(possible_actions))
        # input = parse(Int, readline())
        # GI.play!(env, input)
    end
    return spec, env
end

# spec, env = run_test()