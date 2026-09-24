"""Tests for the PSL match-centre scraper."""

import unittest

from src.data.sources.psl_scraper import (
    PSLScraperSource,
    _clean_team_name,
)


class TestPSLScraperSource(unittest.TestCase):

    def test_clean_team_name(self):
        self.assertEqual(_clean_team_name("  Orlando Amstel Arena  "), "Orlando Pirates")
        self.assertEqual(_clean_team_name("AmaZulu FC"), "Amazulu")
        self.assertEqual(_clean_team_name("  Milford FC  "), "Milford")

    def test_parse_team(self):
        html = (
            '<td class="fixtures-team1">'
            '<div class="team-meta">'
            '<h6 class="team-meta__name">  Kaizer Chiefs  FC </h6>'
            '</div></td>'
        )
        src = PSLScraperSource()
        self.assertEqual(src._parse_team(html, "fixtures-team1"), "Kaizer Chiefs")

    def test_parse_score(self):
        html = '<td class="results-score">\n<span> 2 - 1 </span>\n</td>'
        src = PSLScraperSource()
        self.assertEqual(src._parse_score(html), (2, 1))

    def test_parse_score_returns_none_for_scheduled(self):
        html = '<td class="fixtures-vs">VS</td>'
        src = PSLScraperSource()
        self.assertIsNone(src._parse_score(html))

    def test_parse_time_with_time(self):
        html = '<td colspan="3" class="fixtures-footer-block">13 Sep 15:00 - Sugar Ray Xulu Stadium</td>'
        src = PSLScraperSource()
        dt = src._parse_time(html, "13Sep2026")
        self.assertIsNotNone(dt)
        self.assertIn("2026-09-13T15:00:00", dt.isoformat())

    def test_parse_time_without_time(self):
        html = '<td colspan="3" class="fixtures-footer-block">13 Sep - Sugar Ray Xulu Stadium</td>'
        src = PSLScraperSource()
        dt = src._parse_time(html, "13Sep2026")
        self.assertIsNotNone(dt)

    def test_get_standings_from_sample_matches(self):
        src = PSLScraperSource()
        # Override _fetch_html so the test is offline.
        sample_html = """
        <tbody name="results_13Sep2026">
            <tr class="results-top-row">
                <td class="results-team1"><h6 class="team-meta__name">Team A</h6></td>
                <td class="results-score"><span> 3 - 1 </span></td>
                <td class="results-team2"><h6 class="team-meta__name">Team B</h6></td>
            </tr>
            <tr class="fixtures-footer-row">
                <td colspan="3" class="fixtures-footer-block">13 Sep 15:00 - Venue</td>
            </tr>
        </tbody>
        <tbody name="fixtures_20Sep2026">
            <tr class="fixtures-top-row">
                <td class="fixtures-team1"><h6 class="team-meta__name">Team C</h6></td>
                <td class="fixtures-vs">VS</td>
                <td class="fixtures-team2"><h6 class="team-meta__name">Team D</h6></td>
            </tr>
            <tr class="fixtures-footer-row">
                <td colspan="3" class="fixtures-footer-block">20 Sep 15:00 - Venue</td>
            </tr>
        </tbody>
        """
        src._fetch_html = lambda: sample_html
        src._fetch_log_html = lambda slug: ""  # force the computed-table fallback
        matches = src.get_matches({"psl_scraper": {}})
        self.assertEqual(len(matches), 2)

        standings = src.get_standings({"psl_scraper": {}})
        self.assertEqual(len(standings), 4)
        self.assertEqual(standings[0]["team"], "Team A")
        self.assertEqual(standings[0]["points"], 3)
        self.assertEqual(standings[0]["position"], 1)

    def test_parse_log(self):
        html = """
        <tbody id="LogViewContent">
            <tr>
                <td class="logs-team">
                    <div class="team-meta">
                        <h5 class="team-meta__name">1   </h5>
                        <h6 class="team-meta__name">Mamelodi Sundowns</h6>
                    </div>
                </td>
                <td class="logs-played">8</td>
                <td class="logs-win">6</td>
                <td class="logs-draw">2</td>
                <td class="logs-lost">0</td>
                <td class="logs-goals-for">19</td>
                <td class="logs-goals-against">7</td>
                <td class="logs-goal-diff">12</td>
                <td class="logs-points">20</td>
            </tr>
            <tr>
                <td class="logs-team">
                    <div class="team-meta">
                        <h5 class="team-meta__name">2   </h5>
                        <h6 class="team-meta__name">AmaZulu FC</h6>
                    </div>
                </td>
                <td class="logs-played">7</td>
                <td class="logs-win">5</td>
                <td class="logs-draw">1</td>
                <td class="logs-lost">1</td>
                <td class="logs-goals-for">10</td>
                <td class="logs-goals-against">7</td>
                <td class="logs-goal-diff">3</td>
                <td class="logs-points">16</td>
            </tr>
        </tbody>
        """
        rows = PSLScraperSource._parse_log(html)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["team"], "Mamelodi Sundowns")
        self.assertEqual(rows[0]["position"], 1)
        self.assertEqual(rows[0]["points"], 20)
        # "AmaZulu FC" -> strips FC suffix and applies the AmaZulu name fix.
        self.assertEqual(rows[1]["team"], "Amazulu")


if __name__ == "__main__":
    unittest.main()
