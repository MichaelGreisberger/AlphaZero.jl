import AlphaZero.GI
using StaticArrays

@enum ActionType paika=0 approach=1 withdrawal=2 pass=3
@enum BoardSize small=9 medium=25 large=45

### CONSTANTS
const EMPTY = 0x00
const WHITE = 0x01
const BLACK = 0x02

### TYPES
const Player = UInt8
const Cell = UInt8
const Coord = UInt8
const Direction = UInt8
const Board = Array{Cell}

### DIFFERENT INITIAL BOARDS
const INITIAL_BOARD_LARGE = MVector{45, Cell}(
    BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,WHITE,BLACK,WHITE,EMPTY,BLACK,WHITE,BLACK,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE,WHITE
)

const INITIAL_BOARD_MEDIUM = MVector{25, Cell}(
    BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,BLACK,BLACK,BLACK,BLACK,
    BLACK,WHITE,EMPTY,BLACK,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE,
    WHITE,WHITE,WHITE,WHITE,WHITE
)

const INITIAL_BOARD_SMALL = MVector{9, Cell}(
    BLACK,BLACK,BLACK,
    BLACK,EMPTY,WHITE,
    WHITE,WHITE,WHITE
)

#######################################
### IMPLEMENTATION OF GAMEINTERFACE ###
#######################################

"""
Structure to store information related to an action
"""
struct Action
    type :: ActionType
    x0 :: Coord
    y0 :: Coord
    x1 :: Coord
    y1 :: Coord
    direction :: Direction
end

"""
Implementation of abstract type for a game specification.

The specification holds all _static_ information about a game, which does not
depend on the current state.
"""
struct GameSpec <: GI.AbstractGameSpec 
    possible_actions :: Array{Action}
    action_indices :: Dict{Action, Int}
    size :: BoardSize
    num_cols :: UInt8
    num_rows :: UInt8
    num_cells :: UInt8
    pieces_per_player :: UInt8
    num_actions :: Int
    initial_board :: Board

    function GameSpec()
        return GameSpec(large)
    end

    function GameSpec(size::BoardSize)
        num_cols = size == small ? 0x03 : size == medium ? 0x05 : 0x09
        num_rows = size == small ? 0x03 : 0x05
        num_cells = num_cols * num_rows
        pieces_per_player = UInt8(floor(num_cells/2))
        initial_board = size == small ? INITIAL_BOARD_SMALL : size == medium ? INITIAL_BOARD_MEDIUM : INITIAL_BOARD_LARGE
        possible_actions = calc_possible_actions(num_cols, num_rows)
        action_indices = calc_action_indices(possible_actions)
        num_actions = length(possible_actions)
        return new(possible_actions, action_indices, size, num_cols, num_rows, num_cells, pieces_per_player, num_actions, initial_board)        
    end
end

"""
Implementation of abstract base type for a game environment.

Intuitively, a game environment holds a game specification and a current state.
"""
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

"""
Create a new game environment in an initial state depending on the size specified.
"""
function GI.init(spec::GameSpec)
    board = copy(spec.initial_board)
    current_player = WHITE
    env = GameEnv(spec, board, current_player, spec.pieces_per_player, spec.pieces_per_player, [], false, (0x0, 0x0), [], 0)
    update_actions_mask!(env)
    return env
end

"""
Return the game specification of an environment.
"""
GI.spec(g::GameEnv) = g.spec

#####
##### Queries on specs
#####

"""
Return whether or not a game is a two-players game.
"""
GI.two_players(::GameSpec) = true

"""
Return the vector of all game actions.
"""
GI.actions(spec::GameSpec) = spec.possible_actions

"""
Return a vectorized representation of a given state.
"""
function GI.vectorize_state(spec::GameSpec, state)
    board = state.current_player == WHITE ? state.board : flip_colors(spec.num_cells, state.board)
    return Float32[
        board[position]
        for position in 1:spec.num_cells
    ]
end

#####
##### Operations on envs
#####

"""
Modify the state of a game environment in place.
"""
function GI.set_state!(g::GameEnv, state)
    g.board = copy(state.board)
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

"""
Return the game state.
"""
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

"""
Return a boolean indicating whether or not the game is in a terminal state.
"""
function GI.game_terminated(g::GameEnv)
    return !(g.white_pieces > 0 && g.black_pieces > 0)
end

"""
Return `true` if white is to play and `false` otherwise.
"""
function GI.white_playing(g::GameEnv)
    return g.current_player == WHITE
end

"""
Return a boolean mask indicating what actions are available.

The following identities must hold:

  - `game_terminated(game) || any(actions_mask(game))`
  - `length(actions_mask(game)) == length(actions(spec(game)))`
"""
function GI.actions_mask(g::GameEnv)
    return g.action_mask
end

"""
Update the game environment by making the current player perform `action`.
Note that this function does not have to be deterministic.
"""
function GI.play!(g::GameEnv, action::Action)
    if action.type == pass
        swap_player!(g)
    elseif action.type == paika
        apply_action!(g, action)
        swap_player!(g)
    else
        apply_action!(g, action)

        g.extended_capture = true
        g.last_direction = action.direction
        push!(g.last_positions, (action.x0, action.y0))
        g.current_position = (action.x1, action.y1)

        found_capture = update_actions_mask_extended_capture!(g)
        
        if !found_capture
            swap_player!(g)
        end
    end
end

"""
Return the intermediate reward obtained by the white player after the last
transition step. The result is undetermined when called at an initial state.
"""
function GI.white_reward(g::GameEnv)
    if GI.game_terminated(g)
        return g.white_pieces > 0.0 ? 1.0 : -1.0
    else
        return 0
    end
end

"""
Return a heuristic estimate of the state value for the current player.

The given state must be nonfinal and returned values must belong to the
``(-∞, ∞)`` interval.

This function is not needed by AlphaZero but it is useful for building
baselines such as minmax players.
"""
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

"""
Print the game state on the standard output.
"""
function GI.render(g::GameEnv)
    board = g.board
    buffer = IOBuffer()

    xLength = g.spec.num_cols * 02 - 01
    yLength = g.spec.num_rows * 02 - 01

    player = g.current_player == WHITE ? 'W' : 'B'

    if g.spec.size == small
        println(buffer, player * " a   b   c")
    elseif g.spec.size == medium
        println(buffer, player * " a   b   c   d   e")
    else
        println(buffer, player * " a   b   c   d   e   f   g   h   i")
    end

    for y in UnitRange(1, yLength)
        print(buffer, y % 2 != 0 ? string(ceil(Int, y / 2)) * " " : "  ")
        for x in UnitRange(1, xLength)
            if x % 2 != 0
                if y % 2 != 0
                    pos = cord_to_pos(g.spec, floor(UInt8, x/2 + 1), floor(UInt8, y/2 + 1))
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

"""
Return a human-readable string representing the provided action.
"""
function GI.action_string(::GameSpec, action::Action)
    if action.type == pass
        return "pass"
    else
        return Char(action.x0 + 96) * string(action.y0) * Char(action.x1 + 96) * string(action.y1) * (action.type == paika ? 'P' : action.type == approach ? 'A' : action.type == withdrawal ? 'W' : error("Unknown action type $(action.type)"))
    end
end

"""
Return the action described by string `str` or `nothing` if `str` does not
denote a valid action.
"""
function GI.parse_action(::GameSpec, str::String)
    if str == "pass"
        return Action(pass, 0x00, 0x00, 0x00, 0x00, 0x00)
    else
        parts = collect(Char, str)
        x0 = Coord(parts[1] - 96)
        y0 = Coord(parse(Int, parts[2]))
        x1 = Coord(parts[3] - 96)
        y1 = Coord(parse(Int, parts[4]))
        d = determine_direction(x0, y0, x1, y1)
        return Action(x0, y0, x1, y1, d, paika : parts[5] == 'A' ? approach : withdrawal)
    end
end

"""
Read a state from the standard input.
Return the corresponding state (with type `state_type(game_spec)`)
or `nothing` in case of an invalid input.
"""
function GI.read_state(spec::GameSpec)
    board = Board(undef, NUM_CELLS)
    println(board)
    try
        i = 1
        input = lowercase(readline())
        player = input[1] == 'w' ? WHITE : input[1] == 'b' ? BLACK : error("First character must either be 'W' or 'B' to identify the current player")
        linecount = spec.size == small ? 6 : 10
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

"""
Calculates the opponent of given player and returns it
"""
other(p::Player) = 0x03 - p

"""
Applies an action inplace to given GameEnv. Is capable of applying any simple action. 
"""
function apply_action!(g::GameEnv, action::Action)
    if action.type == approach
        dirX = action.x1 - action.x0
        dirY = action.y1 - action.y0
        capture!(g, action.x1 + dirX, action.y1 + dirY, dirX, dirY, other(g.current_player))
    elseif action.type == withdrawal
        dirX = action.x0 - action.x1
        dirY = action.y0 - action.y1
        capture!(g, action.x1 + 0x02 * dirX, action.y1 + 0x02 * dirY, dirX, dirY, other(g.current_player))
    end
    setindex!(g.board, EMPTY, cord_to_pos(g.spec, action.x0, action.y0))
    setindex!(g.board, g.current_player, cord_to_pos(g.spec, action.x1, action.y1))
end

"""
Changes the current player, updates and resets all relevant datastructures inplace.
"""
function swap_player!(g::GameEnv)
    g.current_player = other(g.current_player)
    g.extended_capture = false
    g.last_positions = []
    g.last_direction = 0
    g.current_position = (0x0, 0x0)
    update_actions_mask!(g)
end

"""
Applies a capturing move to GameEnv inplace
"""
function capture!(g::GameEnv, x, y, dirX, dirY, opponent::Player)
    setindex!(g.board, 0x00, cord_to_pos(g.spec, x, y))
    if opponent == WHITE
        g.white_pieces -= 1
    else
        g.black_pieces -= 1
    end

    x1 = x + dirX
    y1 = y + dirY

    if is_in_board_space(g.spec, x1, y1) && g.board[cord_to_pos(g.spec, x1, y1)] == opponent
        capture!(g, x1, y1, dirX, dirY, opponent)
    end
end

"""
Flips the value of a single board cell.
"""
flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

"""
Flips the values of the whole board.
"""
function flip_colors(num_cells::UInt8, board::Board)
    return Cell[
        flip_cell_color(board[position])
        for position in 1:num_cells
    ]
end

"""
Calculates all possible actions for the current board.
"""
function calc_possible_actions(num_cols::UInt8, num_rows::UInt8)
    actions = Action[]

    for x0 in 0x01:num_cols, y0 in 0x01:num_rows
        for (x1, y1, direction) in get_surrounding_nodes(num_cols, num_rows, x0, y0)
            push!(actions, Action(paika, x0, y0, x1, y1, direction))
            push!(actions, Action(approach, x0, y0, x1, y1, direction))
            x_w, y_w = new_coords(x0, y0, opposite_direction(direction)) # calculate position where opponent would reside when withdrawing
            if is_in_board_space(num_cols, num_rows, x_w, y_w) # if the opponents position would be on the board then add this possible move
                push!(actions, Action(withdrawal, x0, y0, x1, y1, direction))
            end
        end
        # x1, y1 = new_coords(x0, y0, direction)
        # # if is_in_board_space(x1, y1)            
        # index = action_index(position, direction)
        # actions[index + UInt8(paika)] = (paika, x0, y0, x1, y1, direction)
        # actions[index + UInt8(approach)] = (approach, x0, y0, x1, y1, direction)
        # actions[index + UInt8(withdrawal)] = (withdrawal, x0, y0, x1, y1, direction)
        # end
    end
    push!(actions, Action(pass, 0x00, 0x00, 0x00, 0x00, 0x00))
    return actions
end

function calc_action_indices(actions::Array{Action})
    indices = Dict{Action, Int}()
    index = 1

    for action in actions
        indices[action] = index
        index += 1
    end

    return indices
end

function action_index(spec::GameSpec, position::UInt8, direction::Direction, type::ActionType)
    x, y = pos_to_cord(spec, position)
    return action_index(spec, x, y, direction, type)
end 

function action_index(spec::GameSpec, x::Coord, y::Coord, direction::Direction, type::ActionType)
    x1, y1 = new_coords(x, y, direction)
    action = Action(type, x, y, x1, y1, direction)
    return spec.action_indices[action]
end

"""
Calculates the destination coordinates of an action originating in (x0, y0) in direction
"""
function new_coords(x0::Coord, y0::Coord, direction::Direction)
    x1 = Int # destination coordinate x (columns)
    y1 = Int # destination coordinate y (rows)

    if direction == 0x01
        x1 = x0
        y1 = y0 - 0x01
    elseif direction == 0x02
        x1 = x0 + 0x01
        y1 = y0 - 0x01
    elseif direction == 0x03
        x1 = x0 + 0x01
        y1 = y0
    elseif direction == 0x04
        x1 = x0 + 0x01
        y1 = y0 + 0x01
    elseif direction == 0x05
        x1 = x0
        y1 = y0 + 0x01
    elseif direction == 0x06
        x1 = x0 - 0x01
        y1 = y0 + 0x01
    elseif direction == 0x07
        x1 = x0 - 0x01
        y1 = y0
    elseif direction == 0x08
        x1 = x0 - 0x01
        y1 = y0 - 0x01
    else
        error("Direction out of bounds (0x01:0x08)")
    end

    return x1, y1
end

"""
Determines the direction according to the origin (x0, y0) and destination (x1, y1)
"""
function determine_direction(x0::Coord, y0::Coord, x1::Coord, y1::Coord)
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

"""
Converts position to x, y coordinates
"""
function pos_to_cord(spec::GameSpec, position::UInt8)
    return pos_to_cord(spec.num_cols, position)
end

"""
Converts position to x, y coordinates
"""
function pos_to_cord(num_cols::UInt8, position::UInt8)
    x = (position - 0x01) % num_cols + 0x01 # origin coordinate x (columns)
    y = UInt8(ceil(position / num_cols)) # origin coordinate y (rows)
    return x, y
end

"""
Converts x, y coordinates to position
"""
function cord_to_pos(spec::GameSpec, x::Coord, y::Coord)
    return UInt8((y - 0x01) * spec.num_cols + x)
end

"""
Updates the action_mask array inplace according to the current board state.
"""
function update_actions_mask!(g::GameEnv)
    g.action_mask = falses(g.spec.num_actions)

    found_capture = false
    paika_moves = Vector{Tuple}()

    player = g.current_player
    opponent = other(player)
    
    for x = UnitRange(0x01, g.spec.num_cols), y = UnitRange(0x01, g.spec.num_rows)
        if g.board[cord_to_pos(g.spec, x, y)] == player
            for (x1, y1, d) in get_empty_neighbours(g.spec, x, y, g.board)
                found_capture |= update_action_mask_for_neighbour!(g, x, y, x1, y1, d, opponent)
                if !found_capture
                    push!(paika_moves, (x, y, d))
                end
            end
        end
    end

    if !found_capture
        for (x, y, d) in paika_moves
            g.action_mask[action_index(g.spec, x, y, d, paika)] = true
        end
    end
    
    return found_capture
end

"""
Updates the action_mask array inplace according to the current board state with respect to an ongoing
extended capture move. 
"""
function update_actions_mask_extended_capture!(g::GameEnv)
    g.action_mask = falses(g.spec.num_actions)
    opponent = other(g.current_player)
    found_capture = false

    for (x1, y1, d) in get_empty_neighbours(g.spec, g.current_position[1], g.current_position[2], g.board)
        if d != g.last_direction && !((x1, y1) ∈ g.last_positions)
            found_capture |= update_action_mask_for_neighbour!(g, g.current_position[1], g.current_position[2], x1, y1, d, opponent)
        end
    end

    g.action_mask[g.spec.num_actions] = true #pass
    
    return found_capture
end

"""
Updates the action_mask array inplace for a specific neighbour node.
"""
function update_action_mask_for_neighbour!(g::GameEnv, x0::Coord, y0::Coord, x1::Coord, y1::Coord, d::Direction, opponent::Player)
    found_capture = false;
    if opponent_in_direction(g, x1, y1, d, opponent)
        found_capture = true;
        g.action_mask[action_index(g.spec, x0, y0, d, approach)] = true
    end
    if opponent_in_direction(g, x0, y0, opposite_direction(d), opponent) 
        found_capture = true
        g.action_mask[action_index(g.spec, x0, y0, d, withdrawal)] = true
    end
    return found_capture
end

"""
Changes given direction to the opposite
"""
function opposite_direction(d::Direction)
    return (d + 0x03) % 0x08 + 0x01
end

"""
Enumerates all neighbours of given position (x, y)
"""
function get_empty_neighbours(spec::GameSpec, x::Coord, y::Coord, board::Board)
    empty_neighbours = Vector{Tuple}()
    for (x1, y1, d) in get_surrounding_nodes(spec, x, y) 
        if board[cord_to_pos(spec, x1, y1)] == EMPTY
            push!(empty_neighbours, (x1, y1, d))
        end
    end
    return empty_neighbours
end

"""
Enumerates all surrounding nodes that are reachable from given position (x, y)
"""
function get_surrounding_nodes(spec::GameSpec, x::Coord, y::Coord)
    return get_surrounding_nodes(spec.num_cols, spec.num_rows, x, y)
end

"""
Enumerates all surrounding nodes that are reachable from given position (x, y)
"""
function get_surrounding_nodes(num_cols::UInt8, num_rows::UInt8, x::Coord, y::Coord)
    if (is_strong_node(num_cols, x, y))
        return get_eight_neighbours(num_cols, num_rows, x, y)
    else
        return get_four_neighbours(num_cols, num_rows, x, y)
    end
end

"""
Indicates whether given position (x, y) is a strong node (strong: eight reachable neighbours, weak: four reachable neighbours)
"""
function is_strong_node(spec::GameSpec, x::Coord, y::Coord)
    return is_strong_node(spec.num_cols, x, y)
end

"""
Indicates whether given position (x, y) is a strong node (strong: eight reachable neighbours, weak: four reachable neighbours)
"""
function is_strong_node(num_cols::UInt8, x::Coord, y::Coord)
    return (x + y * num_cols) % 2 == 0
end

"""
Returns up to eight neighbours depending on whether they reside in the current boards space
"""
function get_eight_neighbours(spec::GameSpec, x::Coord, y::Coord)
    return get_eight_neighbours(spec.num_cols, spec.num_rows, x, y)
end

"""
Returns up to eight neighbours depending on whether they reside in the current boards space
"""
function get_eight_neighbours(num_cols::UInt8, num_rows::UInt8, x::Coord, y::Coord)
    neighbours = Tuple[]
    if (is_in_board_space(num_cols, num_rows, x, y - 0x01))
        push!(neighbours, (x, y - 0x01, 0x01))
    end
    if (is_in_board_space(num_cols, num_rows, x + 0x01, y - 0x01))
        push!(neighbours, (x + 0x01, y - 0x01, 0x02))
    end
    if (is_in_board_space(num_cols, num_rows, x + 0x01, y))
        push!(neighbours, (x + 0x01, y, 0x03))
    end
    if (is_in_board_space(num_cols, num_rows, x + 0x01, y + 0x01))
        push!(neighbours, (x + 0x01, y + 0x01, 0x04))
    end
    if (is_in_board_space(num_cols, num_rows, x, y + 0x01))
        push!(neighbours, (x, y + 0x01, 0x05))
    end
    if (is_in_board_space(num_cols, num_rows, x - 0x01, y + 0x01))
        push!(neighbours, (x - 0x01, y + 0x01, 0x06))
    end
    if (is_in_board_space(num_cols, num_rows, x - 0x01, y))
        push!(neighbours, (x - 0x01, y, 0x07))
    end
    if (is_in_board_space(num_cols, num_rows, x - 0x01, y - 0x01))
        push!(neighbours, (x - 0x01, y - 0x01, 0x08))
    end
    return neighbours
end

"""
Returns up to four neighbours depending on whether they reside in the current boards space
"""
function get_four_neighbours(spec::GameSpec, x::Coord, y::Coord)
    return get_four_neighbours(spec.num_cols, spec.num_rows, x, y)
end

"""
Returns up to four neighbours depending on whether they reside in the current boards space
"""
function get_four_neighbours(num_cols::UInt8, num_rows::UInt8, x::Coord, y::Coord)
    neighbours = Tuple[]
    if (is_in_board_space(num_cols, num_rows, x, y - 0x01))
        push!(neighbours, (x, y - 0x01, 0x01))
    end
    if (is_in_board_space(num_cols, num_rows, x + 0x01, y))
        push!(neighbours, (x + 0x01, y, 0x03))
    end
    if (is_in_board_space(num_cols, num_rows, x, y + 0x01))
        push!(neighbours, (x, y + 0x01, 0x05))
    end
    if (is_in_board_space(num_cols, num_rows, x - 0x01, y))
        push!(neighbours, (x - 0x01, y, 0x07))
    end
    return neighbours
end

"""
Indicates whether given coordinates are within the boundries of the current board
"""
function is_in_board_space(num_cols::UInt8, num_rows::UInt8, x::Coord, y::Coord)    
    return x > 0 && x <= num_cols && y > 0 && y <= num_rows
end

"""
Indicates whether given coordinates are within the boundries of the current board
"""
function is_in_board_space(spec::GameSpec, x::Coord, y::Coord)    
    return is_in_board_space(spec.num_cols, spec.num_rows, x, y)
end

"""
Determines whether an opponent piece resides in given direction
"""
function opponent_in_direction(g::GameEnv, x::Coord, y::Coord, d::Direction, opponent::Player)
    x1, y1 = new_coords(x, y, d)
    
    if is_in_board_space(g.spec, x1, y1)
        return g.board[cord_to_pos(g.spec, x1, y1)] == opponent
    else
        return false
    end
end

"""
Calculates a scalar value that describes how favorable the current situation is for the white player.
This value is a simple material function (white pieces - black pieces) 
"""
function white_heuristic_value(g::GameEnv)
    return Float64(g.white_pieces - g.black_pieces)
end

"""
prints the current state and all possible actions in it.
"""
function print_state_information(g::GameEnv)
    println("state: $(GI.current_state(g))")
    print_possible_action_strings(g)
end

"""
Prints all possible actions of the current state.
"""
function print_possible_action_strings(g::GameEnv)
    print("possible actions: ")
    for action in GI.actions(g.spec)[g.action_mask]
        print(GI.action_string(g.spec, action) * " ($action), ")
    end
    println()
end

####################
### TEST METHODS ###
####################

function run_test()
    spec = GameSpec()
    return run_test(spec)
end

function run_test(spec::GameSpec)
    env = GI.init(spec)
    i = 0
    while !GI.game_terminated(env)
        i += 1 
        # println(GI.render(env))
        # println("its player $(env.current_player == WHITE ? " white's " : " black's ") turn!")
        # print_possible_action_strings(env)
        possible_actions = GI.actions(spec)[env.action_mask]
        GI.play!(env, rand(possible_actions))
        # input = parse(Int, readline())
        # GI.play!(env, input)
    end
    return i, spec, env
end

function performance_test(n::Int)
    for i in 1:n
        run_test()
    end
end