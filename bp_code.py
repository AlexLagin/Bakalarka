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


def build_result_text(eq_length):
    start1 = g1_data.get("start")
    rules1 = g1_data.get("rules_lines", [])
    start2 = g2_data.get("start")
    rules2 = g2_data.get("rules_lines", [])

    lines = []

    if not start1 or not rules1:
        lines.append("Gramatika G1 nie je úplne zadaná.")
    if not start2 or not rules2:
        lines.append("Gramatika G2 nie je úplne zadaná.")
    if eq_length is None:
        lines.append("L_test nie je zadané alebo je neplatné číslo.")

    if lines:
        return "\n".join(lines)

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


### GUI NASTAVENIA ###

BG_COLOR = '#d0e7f9'
TEXT_COLOR = '#00274d'
BUTTON_BG = '#00509e'
BUTTON_FG = 'white'
TITLE_FONT = ("Arial", 20, "bold")
LABEL_FONT = ("Arial", 14)
ENTRY_FONT = ("Arial", 14)
BUTTON_FONT = ("Arial", 16)


def setup_start_frame(frame):
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=0)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Testovanie ekvivalencie dvoch bezkontextových gramatík",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(40, 10))

    tk.Button(
        frame,
        text="Start",
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=18,
        height=2,
        command=lambda: (reset_all_user_inputs(), show_frame(frame_input), g1_inputs["start"].focus_set())
    ).grid(row=1, column=0, pady=10)


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
    # kratšie
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
    # kratšie
    g2_rules = tk.Text(g2_frame, font=ENTRY_FONT, height=8, wrap="word")
    g2_rules.grid(row=1, column=1, pady=5, padx=10, sticky="nsew")

    # uložíme referencie pre reset
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
        g1_data["start"] = g1_start.get().strip()
        rules1_text = g1_rules.get("1.0", tk.END).strip()
        g1_data["rules_lines"] = rules1_text.split("\n") if rules1_text else []

        g2_data["start"] = g2_start.get().strip()
        rules2_text = g2_rules.get("1.0", tk.END).strip()
        g2_data["rules_lines"] = rules2_text.split("\n") if rules2_text else []

        eq_text = entry_eq.get().strip()
        eq_length = None
        if eq_text:
            try:
                eq_length = int(eq_text)
                if eq_length < 0:
                    eq_length = None
            except ValueError:
                eq_length = None

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
        command=lambda: show_frame(frame_input),
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
# zmenšená výška okna
root.geometry("1250x650")
root.minsize(1100, 600)
root.configure(bg=BG_COLOR)

container = tk.Frame(root, bg=BG_COLOR)
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

frame_start = tk.Frame(container, bg=BG_COLOR)
frame_input = tk.Frame(container, bg=BG_COLOR)
frame_result = tk.Frame(container, bg=BG_COLOR)

for f in (frame_start, frame_input, frame_result):
    f.grid(row=0, column=0, sticky="nsew")

setup_start_frame(frame_start)
setup_input_frame(frame_input)
setup_result_frame(frame_result)

show_frame(frame_start)
root.mainloop()
