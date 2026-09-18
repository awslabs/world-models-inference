// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg font-semibold tracking-wide transition-all duration-200 cursor-pointer disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'border-2 border-white/20 text-white bg-transparent hover:border-game-accent hover:bg-game-accent/10',
        primary: 'border-2 border-game-accent bg-game-accent/15 text-white hover:bg-game-accent/30',
        glow: 'border-2 border-game-purple/50 bg-game-purple/10 text-white hover:border-game-purple/80 hover:bg-game-purple/20 hover:shadow-[0_0_20px_rgba(168,85,247,0.2)]',
        ghost: 'text-white/50 hover:text-white hover:bg-white/5',
        danger: 'border border-white/15 bg-white/8 text-white/70 hover:bg-red-500/15 hover:border-red-500/40 hover:text-white',
      },
      size: {
        default: 'h-10 px-6 text-sm',
        sm: 'h-8 px-4 text-xs',
        lg: 'h-12 px-8 text-base',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
