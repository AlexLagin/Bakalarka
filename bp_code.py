import tkinter as tk
import itertools
import re

# Uloženie vstupov pre G1 a G2 medzi scénami
g1_data = {}
g2_data = {}

# referencie na vstupné polia, aby sa dali resetnúť
g1_inputs = {}
g2_inputs = {}
common_inputs = {}  # napr. L_test


### GUI NASTAVENIA ###

BG_COLOR = '#d0e7f9'
TEXT_COLOR = '#00274d'
BUTTON_BG = '#00509e'
BUTTON_FG = 'white'
TITLE_FONT = ("Arial", 20, "bold")
LABEL_FONT = ("Arial", 14)
ENTRY_FONT = ("Arial", 14)
BUTTON_FONT = ("Arial", 16)


### VŠEOBECNÉ FUNKCIE ###

def show_frame(frame):
    frame.tkraise()


def _clear_widget(w):
    if w is None:
        return
    if isinstance(w, tk.Entry):
        w.delete(0, tk.END)
    elif isinstance(w, tk.Text):
        w.delete("1.0", tk.END)


def reset_all_user_inputs():
    g1_data.clear()
    g2_data.clear()

    for w in g1_inputs.values():
        _clear_widget(w)
    for w in g2_inputs.values():
        _clear_widget(w)
    for w in common_inputs.values():
        _clear_widget(w)

    if "result_text_output" in globals():
        try:
            result_text_output.config(state="normal")
            result_text_output.delete("1.0", tk.END)
            result_text_output.config(state="disabled")
            if "result_update_scrollbar" in globals():
                result_update_scrollbar()
        except Exception:
            pass


def normalize_start_symbol(s: str) -> str:
    # Počiatočný symbol je case-insensitive -> uložíme ako veľké
    return (s or "").strip().upper()


def validate_and_parse_eq_length(eq_text: str):
    """
    L_test:
      - prázdne -> 0
      - inak musí obsahovať iba číslice (celé číslo >= 0)
      - ak je neplatné, vráti error text
    """
    t = (eq_text or "").strip()
    if not t:
        return 0, None
    if re.fullmatch(r"\d+", t):
        return int(t), None
    return 0, "• L_test môže byť prázdne alebo musí byť celé číslo (iba číslice 0–9)."


# LHS neterminál: veľké písmená, prípadne apostrofy na konci (S, A, AB, S', A'')
LHS_NONTERMINAL_PATTERN = re.compile(r"^[A-Z]+'*$")


def show_error_popup(title: str, message: str):
    """
    Vlastné popup okno v štýle aplikácie (nie systémový messagebox).
    """
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.configure(bg=BG_COLOR)
    popup.resizable(False, False)

    popup.transient(root)
    popup.grab_set()

    body = tk.Frame(popup, bg=BG_COLOR, padx=18, pady=14)
    body.pack(fill="both", expand=True)

    tk.Label(
        body,
        text=title,
        font=("Arial", 16, "bold"),
        bg=BG_COLOR,
        fg="#b00020"
    ).pack(anchor="w")

    tk.Label(
        body,
        text=message,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        justify="left",
        wraplength=760
    ).pack(anchor="w", pady=(10, 15))

    btn = tk.Button(
        body,
        text="OK",
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=10,
        height=1,
        command=popup.destroy
    )
    btn.pack(anchor="e")

    popup.bind("<Return>", lambda e: popup.destroy())
    popup.bind("<Escape>", lambda e: popup.destroy())

    popup.update_idletasks()
    w = popup.winfo_width()
    h = popup.winfo_height()
    x = root.winfo_rootx() + (root.winfo_width() // 2) - (w // 2)
    y = root.winfo_rooty() + (root.winfo_height() // 2) - (h // 2)
    popup.geometry(f"{w}x{h}+{x}+{y}")

    popup.focus_set()
    btn.focus_set()
    popup.wait_window()


def collect_rule_syntax_errors(rules_lines):
    """
    Skontroluje formát pravidiel v P poli.
    Každý neprázdny riadok musí mať:
      - obsahovať '->'
      - LHS (pred ->) musí byť VEĽKÝ neterminál (povolené aj S', A'')
      - RHS (za ->) nesmie byť prázdna
      - alternatívy oddelené | nesmú byť prázdne (na epsilon používaj '()')
    Vráti zoznam záznamov: (riadok, raw_line, reason_string)
    """
    errors = []

    for idx, line in enumerate(rules_lines, start=1):
        raw = line.strip()
        if not raw:
            continue

        if "->" not in raw:
            errors.append((idx, raw, "Chýba '->'."))
            continue

        left, right = raw.split("->", 1)
        left = left.strip()
        right = right.strip()

        if not left:
            errors.append((idx, raw, "Chýba ľavá strana pred '->'."))
        elif not LHS_NONTERMINAL_PATTERN.match(left):
            errors.append((idx, raw, "Ľavá strana musí byť VEĽKÝ neterminál (povolené aj S', A'')."))

        if right == "":
            errors.append((idx, raw, "Chýba pravá strana za '->'."))
        else:
            alts = [a.strip() for a in right.split("|")]
            for a in alts:
                if a == "":
                    errors.append((idx, raw, "Prázdna alternatíva za '|'. Pre epsilon použi '()'."))

    return errors


def validate_all_inputs_and_collect_errors(
    start1_raw, rules1_lines,
    start2_raw, rules2_lines
):
    """
    Nazbiera VŠETKY chyby naraz (G1, G2 + pravidlá).
    L_test sa tu nerieši (to riešime zvlášť a pripojíme do popupu).
    """
    errors = []

    # G1 completeness
    if not start1_raw and not rules1_lines:
        errors.append("• Gramatika G1 nie je úplne zadaná (chýba počiatočný symbol aj pravidlá).")
    elif not start1_raw:
        errors.append("• Gramatika G1 nie je úplne zadaná (chýba počiatočný symbol).")
    elif not rules1_lines:
        errors.append("• Gramatika G1 nie je úplne zadaná (chýbajú pravidlá).")

    # G2 completeness
    if not start2_raw and not rules2_lines:
        errors.append("• Gramatika G2 nie je úplne zadaná (chýba počiatočný symbol aj pravidlá).")
    elif not start2_raw:
        errors.append("• Gramatika G2 nie je úplne zadaná (chýba počiatočný symbol).")
    elif not rules2_lines:
        errors.append("• Gramatika G2 nie je úplne zadaná (chýbajú pravidlá).")

    # Syntax of rules (len keď používateľ niečo napísal do P)
    if rules1_lines:
        errs = collect_rule_syntax_errors(rules1_lines)
        if errs:
            preview = "\n".join([f"    Riadok {i}: {txt}\n        → {reason}" for i, txt, reason in errs[:8]])
            more = "" if len(errs) <= 8 else f"\n    ... a ďalších {len(errs) - 8} chýb."
            errors.append("• Pravidlá v G1 sú nesprávne zadané:\n" + preview + more)

    if rules2_lines:
        errs = collect_rule_syntax_errors(rules2_lines)
        if errs:
            preview = "\n".join([f"    Riadok {i}: {txt}\n        → {reason}" for i, txt, reason in errs[:8]])
            more = "" if len(errs) <= 8 else f"\n    ... a ďalších {len(errs) - 8} chýb."
            errors.append("• Pravidlá v G2 sú nesprávne zadané:\n" + preview + more)

    return errors


### FUNKCIE PRE PRÁCU S GRAMATIKOU ###

def process_rules(rules_input):
    """
    Každý riadok: S->aAB | b
    "()" sa interpretuje ako epsilon.
    """
    rules = {}
    for rule in rules_input:
        if "->" in rule:
            left, right = rule.split("->")
            left = left.strip()
            right = [r.strip() for r in right.split("|")]
            rules[left] = ["" if r == "()" else r for r in right]
    return rules


def find_simple_rules(grammar):
    simple_rules = {}
    for A, productions in grammar.items():
        for prod in productions:
            if len(prod) == 1 and prod.isupper():
                simple_rules.setdefault(A, []).append(prod)
    return simple_rules


def remove_simple_rules(grammar, simple_rules):
    new_grammar = {key: set(value) for key, value in grammar.items()}
    changed = True
    while changed:
        changed = False
        for A, B_list in simple_rules.items():
            for B in B_list:
                if B in grammar:
                    for prod in grammar[B]:
                        if prod not in new_grammar[A]:
                            new_grammar[A].add(prod)
                            changed = True
        simple_rules = find_simple_rules(new_grammar)

    for A in list(new_grammar.keys()):
        new_grammar[A] = {p for p in new_grammar[A]
                          if not (len(p) == 1 and p.isupper())}

    return {A: list(v) for A, v in new_grammar.items()}


def canonical_form(prod):
    return re.sub(r"[A-Z](')?", "N", prod)


def merge_equivalent_non_terminals_once(grammar, original_nonterminals):
    reverse_grammar = {}
    for nt, productions in grammar.items():
        canon_prods = sorted(canonical_form(p) for p in productions)
        key = tuple(canon_prods)
        reverse_grammar.setdefault(key, []).append(nt)

    merged_grammar = dict(grammar)
    changed = False

    for nts in reverse_grammar.values():
        if len(nts) > 1:
            candidates = [nt for nt in nts if nt in original_nonterminals]
            winner = candidates[0] if candidates else nts[0]
            for nt in nts:
                if nt == winner:
                    continue
                if nt in merged_grammar:
                    del merged_grammar[nt]
                for A in list(merged_grammar.keys()):
                    merged_grammar[A] = [p.replace(nt, winner) for p in merged_grammar[A]]
            changed = True

    return merged_grammar, changed


def merge_equivalent_non_terminals_fixpoint(grammar, original_nonterminals):
    changed = True
    current = grammar
    while changed:
        new_grammar, changed = merge_equivalent_non_terminals_once(
            current, original_nonterminals
        )
        current = new_grammar
    return current


def find_epsilon_producing(grammar, non_terminals):
    epsilon_nt = set()
    changed = True
    while changed:
        changed = False
        for nt, productions in grammar.items():
            if nt in epsilon_nt:
                continue
            for prod in productions:
                if prod == "":
                    epsilon_nt.add(nt)
                    changed = True
                    break
                else:
                    all_eps = True
                    for ch in prod:
                        if ch in grammar and ch not in epsilon_nt:
                            all_eps = False
                            break
                        elif ch not in grammar:
                            all_eps = False
                            break
                    if all_eps:
                        epsilon_nt.add(nt)
                        changed = True
                        break
    return epsilon_nt


def parse_production(prod):
    return list(prod)


def join_production(symbols):
    return "".join(symbols)


def remove_epsilon_productions(grammar, start_symbol, epsilon_nt):
    new_grammar = {A: set() for A in grammar.keys()}

    for A, productions in grammar.items():
        for p in productions:
            symbols = parse_production(p)
            nullable_positions = [i for i, sym in enumerate(symbols)
                                  if sym in epsilon_nt]

            subsets = itertools.chain.from_iterable(
                itertools.combinations(nullable_positions, r)
                for r in range(len(nullable_positions) + 1)
            )

            for subset in subsets:
                new_symbols = list(symbols)
                for idx in sorted(subset, reverse=True):
                    new_symbols.pop(idx)
                new_p = join_production(new_symbols)
                new_grammar[A].add(new_p)

    for A in list(new_grammar.keys()):
        new_grammar[A].discard("")

    final_grammar = {}
    for A, prod_set in new_grammar.items():
        if prod_set:
            final_grammar[A] = list(prod_set)
    return final_grammar


def create_new_start_symbol_if_epsilon(final_grammar, original_start, epsilon_nt):
    if original_start in epsilon_nt and original_start in final_grammar:
        new_start = original_start + "'"
        final_grammar[original_start] = [p for p in final_grammar[original_start] if p != ""]
        final_grammar[new_start] = [original_start]
        return final_grammar, new_start

    return final_grammar, original_start


def find_neperspektivne(grammar, non_terminals):
    productive = set()
    changed = True
    while changed:
        changed = False
        for nt, productions in grammar.items():
            if nt in productive:
                continue
            for prod in productions:
                symbols = parse_production(prod)
                is_prod = True
                for sym in symbols:
                    if sym in non_terminals and sym not in productive:
                        is_prod = False
                        break
                if is_prod:
                    productive.add(nt)
                    changed = True
                    break
    return set(non_terminals) - productive


def remove_unproductive(grammar, unproductive):
    clean = {}
    for nt, productions in grammar.items():
        if nt in unproductive:
            continue
        valid = []
        for prod in productions:
            if any(u in prod for u in unproductive):
                continue
            valid.append(prod)
        if valid:
            clean[nt] = valid
    return clean


def find_unreachable(grammar, start_symbol, protected=set()):
    if start_symbol not in grammar:
        return set(grammar.keys()) - protected

    reachable = set(protected)
    reachable.add(start_symbol)
    queue = [start_symbol]

    while queue:
        cur = queue.pop()
        if cur not in grammar:
            continue
        for prod in grammar[cur]:
            for nt in grammar.keys():
                if nt in prod and nt not in reachable:
                    reachable.add(nt)
                    queue.append(nt)

    return set(grammar.keys()) - reachable


def remove_unreachable(grammar, unreachable, protected=set()):
    clean = {}
    for nt, productions in grammar.items():
        if nt in unreachable and nt not in protected:
            continue
        valid = []
        for prod in productions:
            if any(u in prod for u in unreachable if u not in protected):
                continue
            valid.append(prod)
        if valid:
            clean[nt] = valid
    return clean


### ĽAVÁ REKURZIA ###

def check_left_recursion(grammar):
    direct = set()
    indirect = set()

    for A, productions in grammar.items():
        for prod in productions:
            if not prod:
                continue
            if prod.startswith(A):
                direct.add(A)
            elif prod[0].isupper() and not prod.startswith(A):
                B = prod[0]
                if leads_leftmost_to_A(A, B, grammar):
                    indirect.add(A)

    return direct, indirect


def leads_leftmost_to_A(current, target, grammar, visited=None):
    if visited is None:
        visited = set()

    if current in visited:
        return False
    visited.add(current)

    for p in grammar.get(current, []):
        if not p:
            continue
        first_sym = p[0]
        if first_sym == target:
            return True
        if first_sym.isupper() and first_sym != target:
            if leads_leftmost_to_A(first_sym, target, grammar, visited.copy()):
                return True

    return False


def remove_direct_left_recursion_for(ordered_nonterminals, grammar, orig_start):
    for nt in ordered_nonterminals:
        if nt not in grammar:
            continue
        prods = grammar[nt]
        alpha = []
        beta = []
        for prod in prods:
            if prod.startswith(nt):
                alpha.append(prod[len(nt):])
            else:
                beta.append(prod)

        if alpha:
            if nt == orig_start and 'Z' not in ordered_nonterminals:
                candidate = "Z"
                ordered_nonterminals.insert(0, candidate)
            else:
                candidate = nt + "'"

            while candidate in grammar:
                candidate += "'"

            grammar[candidate] = []
            new_beta = []
            for b in beta:
                new_beta.append(b)
                new_beta.append(b + candidate)
            grammar[nt] = new_beta

            new_alpha = []
            for a in alpha:
                new_alpha.append(a + candidate)
                new_alpha.append(a)
            grammar[candidate] = new_alpha


def remove_indirect_left_recursion_bottom_up(grammar, ordered_nonterminals, orig_start, direct):
    G = {A: list(prods) for A, prods in grammar.items()}

    if direct:
        remove_direct_left_recursion_for(ordered_nonterminals, G, orig_start)
        direct = False

    for i in reversed(range(len(ordered_nonterminals))):
        Ai = ordered_nonterminals[i]
        if Ai not in G:
            continue
        for j in range(i + 1):
            Aj = ordered_nonterminals[j]
            if Aj not in G:
                continue
            if Aj == Ai:
                continue
            new_prods = []
            for prod in G[Aj]:
                if prod.startswith(Ai):
                    alpha = prod[len(Ai):]
                    for gamma in G[Ai]:
                        new_prods.append(gamma + alpha)
                else:
                    new_prods.append(prod)
            G[Aj] = new_prods

    return G


### GENEROVANIE REŤAZCOV ###

def generate_strings_up_to_length(grammar, start_symbol, max_length):
    if start_symbol not in grammar:
        return []

    nonterminals = set(grammar.keys())
    sorted_nts = sorted(nonterminals, key=len, reverse=True)

    def tokenize(prod):
        tokens = []
        i = 0
        while i < len(prod):
            matched = None
            for nt in sorted_nts:
                if prod.startswith(nt, i):
                    matched = nt
                    break
            if matched is not None:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(prod[i])
                i += 1
        return tokens

    memo_nt = {}
    memo_seq = {}

    def gen_nt(nt, remaining, stack):
        key = (nt, remaining)
        if key in memo_nt:
            return memo_nt[key]
        if key in stack:
            return set()
        stack.add(key)
        results = set()
        for prod in grammar.get(nt, []):
            if prod == "":
                if remaining >= 0:
                    results.add("")
                continue
            tokens = tokenize(prod)
            for s in gen_seq(tokens, remaining, stack):
                if len(s) <= remaining:
                    results.add(s)
        stack.remove(key)
        memo_nt[key] = results
        return results

    def gen_seq(tokens, remaining, stack):
        key = (tuple(tokens), remaining)
        if key in memo_seq:
            return memo_seq[key]
        if remaining < 0:
            return set()
        if not tokens:
            return {""}
        first, *rest = tokens
        results = set()
        if first in nonterminals:
            for s1 in gen_nt(first, remaining, stack):
                rem = remaining - len(s1)
                if rem < 0:
                    continue
                for s2 in gen_seq(rest, rem, stack):
                    results.add(s1 + s2)
        else:
            t = first
            if remaining < len(t):
                memo_seq[key] = set()
                return set()
            for s2 in gen_seq(rest, remaining - len(t), stack):
                results.add(t + s2)
        memo_seq[key] = results
        return results

    all_strings = set()
    for s in gen_nt(start_symbol, max_length, set()):
        if len(s) <= max_length:
            all_strings.add(s)

    return sorted(all_strings, key=lambda x: (len(x), x))


### OPTIMALIZÁCIA ###

def optimize_grammar(start_symbol, rules_input):
    original_grammar = process_rules(rules_input)
    original_non_terminals = list(original_grammar.keys())

    epsilon_nt = find_epsilon_producing(original_grammar, original_non_terminals)
    grammar_eps = remove_epsilon_productions(original_grammar, start_symbol, epsilon_nt)

    grammar_with_start, new_start_symbol = create_new_start_symbol_if_epsilon(
        grammar_eps, start_symbol, epsilon_nt
    )

    grammar_no_simple = remove_simple_rules(
        grammar_with_start, find_simple_rules(grammar_with_start)
    )

    direct = False
    direct_rec, indirect_rec = check_left_recursion(grammar_no_simple)

    if indirect_rec:
        if direct_rec:
            direct = True
        ordered_nts = list(reversed(grammar_no_simple.keys()))
        grammar_left = remove_indirect_left_recursion_bottom_up(
            grammar_no_simple, ordered_nts, start_symbol, direct
        )
    else:
        ordered_nts = list(grammar_no_simple.keys())
        G = {A: list(prods) for A, prods in grammar_no_simple.items()}
        for A in ordered_nts:
            remove_direct_left_recursion_for([A], G, start_symbol)
        grammar_left = G
        direct_rec = set()
        direct = False

    grammar_no_simple = remove_simple_rules(grammar_left, find_simple_rules(grammar_left))
    direct_rec, indirect_rec = check_left_recursion(grammar_no_simple)

    if indirect_rec:
        ordered_nts = list(reversed(grammar_no_simple.keys()))
        if new_start_symbol != start_symbol and new_start_symbol not in ordered_nts:
            ordered_nts.insert(0, new_start_symbol)
        grammar_left = remove_indirect_left_recursion_bottom_up(
            grammar_no_simple, ordered_nts, start_symbol, direct
        )
    else:
        ordered_nts = list(grammar_no_simple.keys())
        G = {A: list(prods) for A, prods in grammar_no_simple.items()}
        for A in ordered_nts:
            remove_direct_left_recursion_for([A], G, start_symbol)
        grammar_left = G
        direct_rec = set()
        direct = False

    protected = {new_start_symbol}
    unproductive = find_neperspektivne(grammar_left, original_non_terminals)
    grammar_prod = remove_unproductive(grammar_left, unproductive)
    unreachable = find_unreachable(grammar_prod, new_start_symbol, protected)
    grammar_reach = remove_unreachable(grammar_prod, unreachable, protected)

    final_grammar = merge_equivalent_non_terminals_fixpoint(
        grammar_reach, original_non_terminals
    )
    return final_grammar, new_start_symbol


def build_result_text(eq_length: int):
    start1 = g1_data.get("start")
    rules1 = g1_data.get("rules_lines", [])
    start2 = g2_data.get("start")
    rules2 = g2_data.get("rules_lines", [])

    lines = []

    # ŠPECIÁLNY PRÍPAD:
    # Ak ani v G1 ani v G2 nie sú zadané pravidlá, zobraz výsledok normálne (bez hlášok o neúplnosti).
    both_rules_empty = (not rules1) and (not rules2)

    if not both_rules_empty:
        if not start1 or not rules1:
            lines.append("Gramatika G1 nie je úplne zadaná.")
        if not start2 or not rules2:
            lines.append("Gramatika G2 nie je úplne zadaná.")

        if lines:
            return "\n".join(lines)

    # Ak sú obe prázdne, tak optimalizované gramatiky budú prázdne a porovnanie prebehne normálne.
    if both_rules_empty:
        final1, start1_opt = {}, start1 or ""
        final2, start2_opt = {}, start2 or ""
    else:
        final1, start1_opt = optimize_grammar(start1, rules1)
        final2, start2_opt = optimize_grammar(start2, rules2)

    lines.append("Optimalizovaná gramatika G1:")
    if final1:
        for lhs, prods in final1.items():
            pstr = " | ".join("ε" if p == "" else p for p in prods)
            lines.append(f"  {lhs} -> {pstr}")
    else:
        lines.append("  (prázdna)")

    lines.append("")
    lines.append("Optimalizovaná gramatika G2:")
    if final2:
        for lhs, prods in final2.items():
            pstr = " | ".join("ε" if p == "" else p for p in prods)
            lines.append(f"  {lhs} -> {pstr}")
    else:
        lines.append("  (prázdna)")

    # porovnanie jazykov do dĺžky eq_length
    if final1:
        strings1 = generate_strings_up_to_length(final1, start1_opt, eq_length)
    else:
        strings1 = [""]

    if final2:
        strings2 = generate_strings_up_to_length(final2, start2_opt, eq_length)
    else:
        strings2 = [""]

    set1 = set(strings1)
    set2 = set(strings2)

    lines.append("")
    if set1 == set2:
        lines.append(f"Výsledok: pre dĺžky ≤ {eq_length} sú jazyky G1 a G2 ekvivalentné.")
    else:
        lines.append(f"Výsledok: pre dĺžky ≤ {eq_length} NIE sú jazyky G1 a G2 ekvivalentné.")

    return "\n".join(lines)


### GUI SETUP ###

def setup_start_frame(frame):
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=0)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_rowconfigure(3, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Testovanie ekvivalencie dvoch bezkontextových gramatík",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(40, 10))

    # Tlačidlá pod sebou (Start + Info)
    btns = tk.Frame(frame, bg=BG_COLOR)
    btns.grid(row=1, column=0, pady=10)

    tk.Button(
        btns,
        text="Start",
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=18,
        height=2,
        command=lambda: (reset_all_user_inputs(), show_frame(frame_input), g1_inputs["start"].focus_set())
    ).grid(row=0, column=0, padx=10, pady=(0, 10))

    tk.Button(
        btns,
        text="Info",
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=18,
        height=1,
        command=lambda: show_frame(frame_info)
    ).grid(row=1, column=0, padx=10)


def setup_info_frame(frame):
    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Info / Návod",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(15, 5))

    body = tk.Frame(frame, bg=BG_COLOR)
    body.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    body.grid_rowconfigure(0, weight=1)
    body.grid_columnconfigure(0, weight=1)

    info_text = (
        "Tento program testuje ekvivalenciu dvoch bezkontextových gramatík G1 a G2.\n"
        "Ako zadávať gramatiku:\n"
        "1) Počiatočný symbol (S)\n"
        "   • Neterminál nemusí byť zadaný veľkým písmenom \n"
        "   • Povolené sú aj apostrofy, napr. S', A''.\n\n"
        "2) Pravidlá (P -)\n"
        "   • Každý riadok musí mať tvar:\n"
        "       A->α | β | ...\n"
        "   • Alternatívy oddeľuj znakom | (pipe)\n"
        "   • Epsilon (prázdne slovo) zapisuj ako: ()\n"
        "   • Neterminály musia byť veľké písmená\n"
        "   • Terminály môžu byť napr. malé písmená (a, b, c, ...), číslice alebo iné znaky\n\n"
        "3) L_test\n"
        "   • Určuje maximálnu dĺžku reťazcov, do ktorej sa porovnávajú jazyky.\n"
        "   • Prázdne = 0.\n"
        "   • Inak zadaj celé číslo (napr. 5, 10, 12...).\n\n"
    )

    text_output = tk.Text(
        body,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        wrap="word",
        borderwidth=0,
        highlightthickness=0,
    )
    text_output.grid(row=0, column=0, sticky="nsew")
    text_output.insert("1.0", info_text)
    text_output.config(state="disabled")

    scrollbar = tk.Scrollbar(body, orient="vertical", command=text_output.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    text_output.configure(yscrollcommand=scrollbar.set)

    btns = tk.Frame(frame, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(0, 15))

    tk.Button(
        btns,
        text="Späť",
        command=lambda: show_frame(frame_start),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=12,
        height=1,
    ).grid(row=0, column=0, padx=10, pady=5)

    # klávesové skratky
    frame.bind_all("<Escape>", lambda e: show_frame(frame_start))


def setup_input_frame(frame):
    global g1_inputs, g2_inputs, common_inputs

    frame.grid_rowconfigure(0, weight=0)  # title
    frame.grid_rowconfigure(1, weight=1)  # blocks
    frame.grid_rowconfigure(2, weight=0)  # controls
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Zadávanie gramatík",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(15, 5))

    blocks = tk.Frame(frame, bg=BG_COLOR)
    blocks.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    blocks.grid_rowconfigure(0, weight=1)
    blocks.grid_columnconfigure(0, weight=1)
    blocks.grid_columnconfigure(1, weight=1)

    # --- G1 ---
    g1_frame = tk.LabelFrame(blocks, text="Gramatika G1", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
    g1_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
    g1_frame.grid_rowconfigure(1, weight=1)
    g1_frame.grid_columnconfigure(1, weight=1)

    tk.Label(g1_frame, text="S -", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)\
        .grid(row=0, column=0, pady=5, padx=10, sticky="w")
    g1_start = tk.Entry(g1_frame, font=ENTRY_FONT)
    g1_start.grid(row=0, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(g1_frame, text="P -", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)\
        .grid(row=1, column=0, pady=5, padx=10, sticky="nw")
    g1_rules = tk.Text(g1_frame, font=ENTRY_FONT, height=8, wrap="word")
    g1_rules.grid(row=1, column=1, pady=5, padx=10, sticky="nsew")

    # --- G2 ---
    g2_frame = tk.LabelFrame(blocks, text="Gramatika G2", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
    g2_frame.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
    g2_frame.grid_rowconfigure(1, weight=1)
    g2_frame.grid_columnconfigure(1, weight=1)

    tk.Label(g2_frame, text="S -", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)\
        .grid(row=0, column=0, pady=5, padx=10, sticky="w")
    g2_start = tk.Entry(g2_frame, font=ENTRY_FONT)
    g2_start.grid(row=0, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(g2_frame, text="P -", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)\
        .grid(row=1, column=0, pady=5, padx=10, sticky="nw")
    g2_rules = tk.Text(g2_frame, font=ENTRY_FONT, height=8, wrap="word")
    g2_rules.grid(row=1, column=1, pady=5, padx=10, sticky="nsew")

    g1_inputs = {"start": g1_start, "rules": g1_rules}
    g2_inputs = {"start": g2_start, "rules": g2_rules}

    # --- CONTROLS: L_test nad tlačidlami ---
    controls = tk.Frame(frame, bg=BG_COLOR)
    controls.grid(row=2, column=0, pady=(0, 15), sticky="ew")
    controls.grid_columnconfigure(0, weight=1)
    controls.grid_columnconfigure(1, weight=0)
    controls.grid_columnconfigure(2, weight=1)

    middle = tk.Frame(controls, bg=BG_COLOR)
    middle.grid(row=0, column=1)

    tk.Label(middle, text="L_test -", font=LABEL_FONT, bg=BG_COLOR, fg=TEXT_COLOR)\
        .grid(row=0, column=0, padx=(0, 10), pady=(5, 2), sticky="e")

    entry_eq = tk.Entry(middle, font=ENTRY_FONT, width=10)
    entry_eq.grid(row=0, column=1, padx=(0, 10), pady=(5, 2), sticky="w")

    def on_test():
        # --- načítanie vstupov ---
        start1_raw = g1_start.get().strip()
        rules1_text = g1_rules.get("1.0", tk.END).strip()
        rules1_lines = rules1_text.split("\n") if rules1_text else []

        start2_raw = g2_start.get().strip()
        rules2_text = g2_rules.get("1.0", tk.END).strip()
        rules2_lines = rules2_text.split("\n") if rules2_text else []

        eq_length, eq_error = validate_and_parse_eq_length(entry_eq.get())

        # uloženie (start symbol case-insensitive)
        g1_data["start"] = normalize_start_symbol(start1_raw)
        g1_data["rules_lines"] = rules1_lines

        g2_data["start"] = normalize_start_symbol(start2_raw)
        g2_data["rules_lines"] = rules2_lines

        # Ak sú pravidlá prázdne v OBOCH gramatikách -> normálny výsledok,
        # ALE ak je L_test neplatné, ukáž popup
        if not rules1_text and not rules2_text:
            if eq_error:
                show_error_popup("Chyby vo vstupe", eq_error)
                return

            text = build_result_text(eq_length)

            result_text_output.config(state="normal")
            result_text_output.delete("1.0", tk.END)
            result_text_output.insert("1.0", text)
            result_text_output.config(state="disabled")
            result_update_scrollbar()

            show_frame(frame_result)
            return

        # Inak: validácia (zobraz všetky chyby naraz)
        all_errors = validate_all_inputs_and_collect_errors(
            start1_raw, rules1_lines,
            start2_raw, rules2_lines
        )

        # pripoj L_test chybu (ak existuje)
        if eq_error:
            all_errors.append(eq_error)

        if all_errors:
            show_error_popup("Chyby vo vstupe", "\n".join(all_errors))
            return

        # OK -> zobraz výsledok
        text = build_result_text(eq_length)

        result_text_output.config(state="normal")
        result_text_output.delete("1.0", tk.END)
        result_text_output.insert("1.0", text)
        result_text_output.config(state="disabled")
        result_update_scrollbar()

        show_frame(frame_result)

    btn_row = tk.Frame(middle, bg=BG_COLOR)
    btn_row.grid(row=1, column=0, columnspan=2, pady=(8, 5))

    tk.Button(
        btn_row,
        text="Testovať ekvivalenciu",
        command=on_test,
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=20,
        height=1,
    ).grid(row=0, column=0, padx=10)

    tk.Button(
        btn_row,
        text="Späť",
        command=lambda: show_frame(frame_start),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=10,
        height=1,
    ).grid(row=0, column=1, padx=10)

    common_inputs = {"eq_len": entry_eq}


def setup_result_frame(frame):
    global result_text_output, result_update_scrollbar

    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Výsledok",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(15, 5))

    output_frame = tk.Frame(frame, bg=BG_COLOR)
    output_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    output_frame.grid_rowconfigure(0, weight=1)
    output_frame.grid_columnconfigure(0, weight=1)

    text_output = tk.Text(
        output_frame,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        wrap="word",
        height=18,
        borderwidth=0,
        highlightthickness=0,
        state="disabled",
    )
    text_output.grid(row=0, column=0, sticky="nsew")

    scrollbar = tk.Scrollbar(output_frame, orient="vertical", command=text_output.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    scrollbar.grid_remove()

    def update_scrollbar():
        text_output.update_idletasks()
        info = text_output.dlineinfo("end-1c")
        if info is None:
            scrollbar.grid_remove()
            return
        y_bottom = info[1] + info[3]
        if y_bottom > text_output.winfo_height():
            scrollbar.grid(row=0, column=1, sticky="ns")
        else:
            scrollbar.grid_remove()

    def yscrollcommand(*args):
        scrollbar.set(*args)

    text_output.configure(yscrollcommand=yscrollcommand)

    def on_output_configure(event):
        update_scrollbar()

    output_frame.bind("<Configure>", on_output_configure)

    btns = tk.Frame(frame, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(0, 15))

    tk.Button(
        btns,
        text="Späť na zadávanie",
        command=lambda: (reset_all_user_inputs(), show_frame(frame_input), g1_inputs["start"].focus_set()),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=18,
        height=1,
    ).grid(row=0, column=0, padx=10, pady=5)

    tk.Button(
        btns,
        text="Na začiatok",
        command=lambda: (reset_all_user_inputs(), show_frame(frame_start)),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=14,
        height=1,
    ).grid(row=0, column=1, padx=10, pady=5)

    result_text_output = text_output
    result_update_scrollbar = update_scrollbar


### HLAVNÉ OKNO ###

root = tk.Tk()
root.title("Testovanie")
root.geometry("1250x650")
root.minsize(1100, 600)
root.configure(bg=BG_COLOR)

container = tk.Frame(root, bg=BG_COLOR)
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

frame_start = tk.Frame(container, bg=BG_COLOR)
frame_info = tk.Frame(container, bg=BG_COLOR)   # ✅ NOVÉ: Info obrazovka
frame_input = tk.Frame(container, bg=BG_COLOR)
frame_result = tk.Frame(container, bg=BG_COLOR)

for f in (frame_start, frame_info, frame_input, frame_result):
    f.grid(row=0, column=0, sticky="nsew")

setup_start_frame(frame_start)
setup_info_frame(frame_info)   # ✅ NOVÉ
setup_input_frame(frame_input)
setup_result_frame(frame_result)

show_frame(frame_start)
root.mainloop()
