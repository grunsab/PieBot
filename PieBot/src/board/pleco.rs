#![cfg(feature = "board-pleco")]
use pleco::{Board as PlecoBoard, MoveList};

pub struct RevBoard {
    board: PlecoBoard,
    stack: Vec<pleco::BitMove>,
}

impl RevBoard {
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        PlecoBoard::from_fen(fen)
            .map(|b| Self {
                board: b,
                stack: Vec::with_capacity(128),
            })
            .map_err(|e| format!("FEN error: {e:?}"))
    }
    pub fn startpos() -> Self {
        Self {
            board: PlecoBoard::start_pos(),
            stack: Vec::with_capacity(128),
        }
    }
    pub fn generate_moves(&self) -> MoveList {
        self.board.generate_moves()
    }
    pub fn make(&mut self, mv: pleco::BitMove) {
        self.board.apply_move(mv);
        self.stack.push(mv);
    }
    pub fn unmake(&mut self) {
        if self.stack.pop().is_some() {
            self.board.undo_move();
        }
    }
    pub fn side_to_move(&self) -> pleco::Player {
        self.board.turn()
    }
    pub fn inner(&self) -> &PlecoBoard {
        &self.board
    }
}
