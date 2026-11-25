// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The sim tasks the praxis-1 cartridge accepts (runner.py DROID_TASKS / YAM_TASKS).
// `prompt` is the task's default instruction in praxis-sim; where the seed picks a
// colour, the prompt names it at run time and `prompt` shows the pattern.

export type PraxisRig = 'droid' | 'yam';

export interface PraxisTask {
  id: string;
  rig: PraxisRig;
  group: 'Everyday' | 'Harder' | 'Two arms';
  label: string;
  prompt: string;
  /** How it went in our own test runs, in words (no numbers that would date). */
  note?: string;
}

const T = (id: string, rig: PraxisRig, group: PraxisTask['group'], label: string, prompt: string, note?: string): PraxisTask =>
  ({ id, rig, group, label, prompt, note });

export const PRAXIS_TASKS: PraxisTask[] = [
  T('lift_cube', 'droid', 'Everyday', 'Lift cube', 'Pick up the red cube.', 'Reliable'),
  T('press_button', 'droid', 'Everyday', 'Press button', 'Press the red button.', 'Reliable'),
  T('block_in_bowl', 'droid', 'Everyday', 'Block in bowl', 'Put the red block in the bowl.'),
  T('stack_block', 'droid', 'Everyday', 'Stack block', 'Put the red block on top of the green block.'),
  T('move_block', 'droid', 'Everyday', 'Move block', 'Move the red block to the <left or right>.'),
  T('cube_on_plate', 'droid', 'Everyday', 'Cube on plate', 'Put the red cube on the blue plate.'),
  T('block_out_of_bowl', 'droid', 'Everyday', 'Block out of bowl', 'Remove the red block from the bowl.'),
  T('two_blocks_in_bowl', 'droid', 'Harder', 'Two blocks in bowl', 'Put the red and the green block in the bowl.', 'Two steps'),
  T('pick_color_block', 'droid', 'Harder', 'Pick the named colour', 'Pick up the <colour> block.', 'Distractors; often misses'),
  T('color_block_in_bowl', 'droid', 'Harder', 'Named colour in bowl', 'Put the <colour> block in the bowl.', 'Distractors; often misses'),
  T('cube_on_color', 'droid', 'Harder', 'Cube on named square', 'Put the red cube on the <colour> square.', 'Distractors; often misses'),
  T('stack_three_blocks', 'droid', 'Harder', 'Stack three blocks', 'Stack the blocks.', 'Two steps; hard'),
  T('stack_color_blocks', 'droid', 'Harder', 'Stack named colours', 'Put the <colour> block on the <colour> block.'),
  T('cube_in_tray', 'droid', 'Harder', 'Cube in tray', 'Put the red cube in the tray.'),
  T('block_in_small_bowl', 'droid', 'Harder', 'Block in small bowl', 'Put the red block in the bowl.'),
  T('cube_on_small_plate', 'droid', 'Harder', 'Cube on small plate', 'Put the red cube on the blue plate.'),
  T('stack_small_block', 'droid', 'Harder', 'Stack on small block', 'Put the red block on top of the green block.'),
  T('peg_in_hole_6mm', 'droid', 'Harder', 'Peg in hole', 'Put the blue peg in the square hole.', 'Precision'),
  T('tennis_ball_in_bag', 'yam', 'Two arms', 'Tennis ball in bag', 'Pick up the tennis ball from the center of the table, then place the tennis ball in the bag', 'Reliable'),
  T('apple_in_bag', 'yam', 'Two arms', 'Apple in bag', 'Pick up the fake apple from the center of the table, then place the fake apple in the bag', 'Reliable'),
  T('dice_in_bag', 'yam', 'Two arms', 'Dice in bag', 'Pick up the foam dice stress ball from the center of the table, then place the foam dice stress ball in the bag', 'Reliable'),
  T('bell_in_bag', 'yam', 'Two arms', 'Bell in bag', 'Pick up the bell from the center of the table, then place the bell in the bag', 'Reliable'),
];

export const PRAXIS_GROUPS: PraxisTask['group'][] = ['Everyday', 'Harder', 'Two arms'];
export const PRAXIS_MAX_SEEDS = 8;
