//! Game-ending conditions under the TaflZero operational contract.
//!
//! Checked in this order after every move: king captured, third occurrence of a
//! position (attackers win), king on a corner, all defenders encircled, exit fort,
//! and finally no legal move for the side to move (it loses). Encirclement is only
//! tested after attacker moves that capture or land next to two attackers, and the
//! fort only after king moves and defender moves near other defenders or the edge;
//! that gating is part of the contract, not just an optimisation.

use crate::geometry::*;
use crate::position::{NO_SQUARE, Position, Side};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reason {
    KingCaptured,
    Repetition,
    KingEscaped,
    Encircled,
    ExitFort,
    NoMoves,
}

impl Reason {
    /// TaflZero's `TerminalType` code (the no-move code depends on who is stuck).
    pub fn taflzero_code(self, loser: Side) -> u8 {
        match self {
            Reason::KingCaptured => 1,
            Reason::Repetition => 2,
            Reason::KingEscaped => 3,
            Reason::Encircled => 4,
            Reason::ExitFort => 5,
            Reason::NoMoves if loser == Side::Attackers => 7,
            Reason::NoMoves => 8,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Outcome {
    pub winner: Side,
    pub reason: Reason,
}

/// Number of occurrences (including this one) at which a position ends the game.
pub const REPETITION_LIMIT: u32 = 3;

impl Position {
    /// The game result in this position, given how many times it has occurred.
    pub fn outcome(&self, occurrences: u32) -> Option<Outcome> {
        let win = |winner, reason| Some(Outcome { winner, reason });
        if !self.has_king() {
            return win(Side::Attackers, Reason::KingCaptured);
        }
        if occurrences >= REPETITION_LIMIT {
            return win(Side::Attackers, Reason::Repetition);
        }
        if CORNERS.has(self.king) {
            return win(Side::Defenders, Reason::KingEscaped);
        }
        if self.encirclement_possible() && self.defenders_encircled() {
            return win(Side::Attackers, Reason::Encircled);
        }
        if self.fort_possible() && self.is_exit_fort() {
            return win(Side::Defenders, Reason::ExitFort);
        }
        if !self.has_legal_move() {
            return win(self.side.other(), Reason::NoMoves);
        }
        None
    }

    /// Gate for the encirclement test: an attacker move that captured or landed next
    /// to at least two other attackers (eight-neighbourhood).
    fn encirclement_possible(&self) -> bool {
        if self.last_to == NO_SQUARE {
            return true;
        }
        if !self.attackers.has(self.last_to) {
            return false;
        }
        self.last_capture || (KING_RING[self.last_to as usize] & self.attackers).count() >= 2
    }

    /// No defender (king included) can reach the edge through non-attacker squares.
    pub fn defenders_encircled(&self) -> bool {
        let reach = Bb::flood(self.defender_side(), !self.attackers);
        (reach & EDGES).is_empty()
    }

    /// Gate for the fort test: any king move, or a defender move landing next to two
    /// defenders or edge squares (one suffices when the defender lands on the edge).
    fn fort_possible(&self) -> bool {
        if self.last_to == NO_SQUARE || self.last_to == self.king {
            return true;
        }
        if !self.defenders.has(self.last_to) {
            return false;
        }
        let ring = KING_RING[self.last_to as usize];
        let nearby = (ring & (self.defenders | EDGES)).count();
        nearby >= 2 || (nearby >= 1 && is_edge(self.last_to))
    }

    /// TaflZero's exit-fort test: the king stands on the edge with a free neighbouring
    /// square, no attacker can reach it, and that stays true after repeatedly removing
    /// every defender that attackers could still conceivably capture.
    ///
    /// A defender counts as capturable when both squares beside it on one line lie in
    /// attacker "areas" (orthogonally connected attacker/empty squares, corners and
    /// throne excluded) of which at least one has an empty square, and two attackers
    /// are available if both squares share an area. Removal only ever enlarges areas,
    /// so removing all capturable defenders per round reaches the same fixed point as
    /// TaflZero's one-at-a-time loop.
    pub fn is_exit_fort(&self) -> bool {
        if !self.has_king() || !is_edge(self.king) {
            return false;
        }
        let king = Bb::from_sq(self.king);
        if (king.neighbors() & !self.occupied()).is_empty() {
            return false;
        }
        let mut defenders = self.defenders;
        loop {
            let reach = Bb::flood(king, !defenders);
            if (reach & self.attackers).any() {
                return false;
            }
            let removable = self.capturable_defenders(defenders);
            if removable.is_empty() {
                return true;
            }
            defenders &= !removable;
        }
    }

    fn capturable_defenders(&self, defenders: Bb) -> Bb {
        let open = !(defenders | self.king_bb() | RESTRICTED);
        let empty = open & !self.attackers;
        // Label each attacker area by its lowest square; record attacker count and
        // whether it holds an empty square.
        let mut area_of = [u8::MAX; NUM_SQUARES];
        let mut areas: Vec<(u32, bool)> = Vec::new();
        let mut unlabeled = self.attackers;
        while unlabeled.any() {
            let seed = Bb::from_sq(unlabeled.first());
            unlabeled &= !seed;
            let area = Bb::flood(seed, open);
            if area.is_empty() {
                // An attacker on a corner or the throne, which only a position set up from a
                // FEN can have, belongs to no area.
                continue;
            }
            let id = areas.len() as u8;
            for sq in area {
                area_of[sq as usize] = id;
            }
            areas.push(((area & self.attackers).count(), (area & empty).any()));
            unlabeled &= !area;
        }
        let pair_capturable = |a: Option<Sq>, b: Option<Sq>| -> bool {
            let (Some(a), Some(b)) = (a, b) else { return false };
            let (ia, ib) = (area_of[a as usize], area_of[b as usize]);
            if ia == u8::MAX || ib == u8::MAX {
                return false;
            }
            if ia == ib && areas[ia as usize].0 < 2 {
                return false;
            }
            areas[ia as usize].1 || areas[ib as usize].1
        };
        let mut removable = Bb::EMPTY;
        for d in defenders {
            // Directions: 0 north, 1 south, 2 east, 3 west.
            if pair_capturable(step(d, 3, 1), step(d, 2, 1)) || pair_capturable(step(d, 0, 1), step(d, 1, 1)) {
                removable.set(d);
            }
        }
        removable
    }
}
