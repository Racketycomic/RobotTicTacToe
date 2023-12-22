# Tic Tac Toe Robot

This repository contains the code and documentation for a Tic Tac Toe game implemented with robotics, computer vision, and minmax algorithms. The project involves human-robot collaboration, where a robot calculates and executes optimal moves using a suction pump to manipulate X and O cubes. The game state is managed through color detection and image processing, with the Minimax algorithm determining strategic moves.

## What I did

- Color-coded Tic Tac Toe cubes (X: Blue, O: Green) manipulated by a suction pump.
- Aruco markers and a grid define the playing area for precise cube positioning.
- Minimax algorithm implementation for strategic move calculation.
- Human and robotic player interactions for dynamic gameplay.
- Real-time image processing to update the game state.

## How I did

1. **Game Setup**
   - Color-coded cubes manipulated with a suction pump.
   - Aruco markers and grid define the playing area.
   - Cube positioning logic based on image coordinates.

2. **Tic Tac Toe Algorithm**
   - Minimax algorithm strategically determines optimal moves.
   - Move evaluation criteria assign scores based on adjacent values.

3. **Gameplay**
   - Human player executes the initial move.
   - Robotic player calculates and executes strategically sound moves.
   - Suction pump transfers cubes, and real positions are calculated.

4. **Game Culmination**
   - Program concludes with a winner declaration or a draw.
   - Final outcome printed in the terminal.

## What I Observed

The project explores functionalities within the OpenCV library, demonstrating image contouring, erosion, masking, and dilation for color detection. The integration of diverse technologies showcases the potential of computer vision in robot applications. The project highlights the adaptability of robots in various workspaces, emphasizing their capacity to perform complex tasks traditionally handled by humans.

## End Result

Watch the gameplay [here](https://www.youtube.com/watch?v=_VxRY-Qkwc4).

**Note:** Detailed information can be found in the [full project report](link-to-full-report).
