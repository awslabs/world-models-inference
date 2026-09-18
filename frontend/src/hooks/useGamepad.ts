// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect, useCallback, useRef } from 'react';

const DEADZONE = 0.2;

function applyDeadzone(value: number): number {
  return Math.abs(value) < DEADZONE ? 0 : value;
}

export interface GamepadAxes {
  /** Left stick: [x, y] each -1..1, 0 = center */
  leftStick: [number, number];
  /** Right stick: [x, y] each -1..1, 0 = center */
  rightStick: [number, number];
}

const ZERO_AXES: GamepadAxes = { leftStick: [0, 0], rightStick: [0, 0] };

export const useGamepad = () => {
  const [gamepadConnected, setGamepadConnected] = useState(false);
  const indexRef = useRef<number | null>(null);

  useEffect(() => {
    const onConnect = (e: GamepadEvent) => {
      console.log(`🎮 Gamepad connected: ${e.gamepad.id}`);
      indexRef.current = e.gamepad.index;
      setGamepadConnected(true);
    };

    const onDisconnect = (_e: GamepadEvent) => {
      console.log('🎮 Gamepad disconnected');
      indexRef.current = null;
      setGamepadConnected(false);
    };

    window.addEventListener('gamepadconnected', onConnect);
    window.addEventListener('gamepaddisconnected', onDisconnect);

    // Check if a gamepad is already connected on mount
    const gamepads = navigator.getGamepads();
    for (const gp of gamepads) {
      if (gp) {
        indexRef.current = gp.index;
        setGamepadConnected(true);
        break;
      }
    }

    return () => {
      window.removeEventListener('gamepadconnected', onConnect);
      window.removeEventListener('gamepaddisconnected', onDisconnect);
    };
  }, []);

  /** Poll current gamepad axes. Call this in your game loop / interval. */
  const pollGamepad = useCallback((): GamepadAxes => {
    if (indexRef.current === null) return ZERO_AXES;

    const gp = navigator.getGamepads()[indexRef.current];
    if (!gp) return ZERO_AXES;

    const axes = gp.axes;
    return {
      leftStick: [
        applyDeadzone(axes[0] ?? 0),
        applyDeadzone(axes[1] ?? 0),
      ],
      rightStick: [
        applyDeadzone(axes[2] ?? 0),
        applyDeadzone(axes[3] ?? 0),
      ],
    };
  }, []);

  return { gamepadConnected, pollGamepad };
};