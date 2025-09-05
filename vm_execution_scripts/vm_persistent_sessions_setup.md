# Setting Up Persistent Sessions on Google Cloud VM

## Problem
When you run a script from VS Code SSH terminal and close VS Code or your computer, the SSH session terminates and kills your running processes. This happens because Google Cloud VMs come with vanilla configurations that don't automatically handle persistent sessions.

## Solution: Auto-start tmux for SSH sessions

### Step 1: Install tmux
```bash
sudo apt update && sudo apt install tmux
```

### Step 2: Add auto-start configuration to bashrc
Run this command to add the configuration:

```bash
echo '
# Auto-start tmux for SSH sessions to keep processes running after disconnect
if [[ -n "$SSH_CLIENT" || -n "$SSH_TTY" ]] && [[ -z "$TMUX" ]]; then
    # Check if there'\''s an existing tmux session
    if tmux list-sessions >/dev/null 2>&1; then
        echo "Existing tmux sessions found:"
        tmux list-sessions
        echo ""
        echo "To attach to a session: tmux attach -t <session-name>"
        echo "To create a new session: tmux new -s <name>"
        echo "To continue in regular shell: just press Enter"
        echo ""
    else
        echo "No existing tmux sessions. Starting new main session..."
        tmux new-session -s main
    fi
fi' >> ~/.bashrc
```

### Step 3: Reload bashrc
```bash
source ~/.bashrc
```

## How It Works

### What the configuration does:
1. **Detects SSH sessions**: Only activates when you're connected via SSH (`$SSH_CLIENT` or `$SSH_TTY` is set)
2. **Checks if already in tmux**: Prevents nested tmux sessions (`$TMUX` is not set)
3. **Lists existing sessions**: Shows you any running tmux sessions when you connect
4. **Auto-creates session**: If no sessions exist, automatically starts a new "main" session

### tmux Key Commands:
- **`Ctrl+B, then D`** = Detach from session (keeps running in background)
- **`Ctrl+B, then C`** = Create new window within session
- **`Ctrl+B, then %`** = Split window vertically
- **`Ctrl+B, then "`** = Split window horizontally

## Usage Workflow

1. **SSH into your VM** - tmux will auto-start or show existing sessions
2. **Run your script normally**: `python run_pretrain.py --sweep`
3. **To disconnect safely**: Press `Ctrl+B`, then `D` (detach)
4. **To reconnect later**: SSH back in and run `tmux attach -t main`

## Benefits

✅ **Scripts keep running** when you disconnect from SSH  
✅ **Automatic setup** - no need to remember to start tmux  
✅ **Multiple sessions** - can run different tasks in different sessions  
✅ **Reconnect anytime** - pick up exactly where you left off  
✅ **Works from any SSH client** - VS Code, terminal, web SSH, etc.

## Alternative: One-time nohup (if you don't want persistent tmux)

If you prefer not to auto-start tmux, you can run individual commands with nohup:

```bash
nohup python run_pretrain.py --sweep > output.log 2>&1 &
disown
```

This runs the script in the background and ignores hangup signals, but you lose the ability to easily reconnect and monitor progress.
