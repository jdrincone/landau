#!/bin/zsh

# Generate a 8-character hexadecimal suffix using openssl
sufix=$(openssl rand -hex 4)
echo "$sufix"