// Code generated by qtc from "organizer.qtpl". DO NOT EDIT.
// See https://github.com/valyala/quicktemplate for details.

package v1

import (
	qtio422016 "io"

	qt422016 "github.com/valyala/quicktemplate"
)

var (
	_ = qtio422016.Copy
	_ = qt422016.AcquireByteBuffer
)

func StreamOrganizerStatic(qw422016 *qt422016.Writer, companies []Company, universities []University) {
	qw422016.N().S(`<!DOCTYPE html>
<html lang="en">

<head>
    <title>Golang companies organizer</title>
    <meta name="description" content="Golang companies organizer">
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="author" type="text/plain" href="https://readytotouch.com/humans.txt"/>
    <meta property="og:image" content="/assets/images/og/organizers-light.jpg">

    `)
	streamfavicon(qw422016)
	qw422016.N().S(`

	<link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap" rel="stylesheet">

    `)
	streamorganizerStyles(qw422016)
	qw422016.N().S(`
</head>

<body>
<header class="header">
	<div class="header__wrapper">
		<a href="/" class="header__logo">
			<img class="header__logo-img" src="/assets/images/pages/online/logo.svg" alt="logo">
			<h3 class="header__logo-title">ReadyToTouch</h3>
		</a>
		<div class="header__stars">
			<iframe src="https://ghbtns.com/github-btn.html?user=readytotouch&repo=readytotouch&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>
		</div>
	</div>
</header>
<main class="main-wrapper">
    <div class="main-container">
        <section class="organizer">
            <div class="wrapper">
                <div class="organizer__table-container">
                    <div class="table__header-top">
                        <p class="table__result-counter">`)
	qw422016.N().D(len(companies))
	qw422016.N().S(` companies</p>
                    </div>
                    <table class="organizer__table table">
                        <thead class="organizer__head">
                            <tr>
                                <th>
                                    <span>Name</span>
                                </th>
                                <th>
                                    <img src="/assets/images/pages/common-images/linkedin.svg" alt="linkedin">
                                    <span>LinkedIn</span>
                                </th>
                                <th>
                                    <img src="/assets/images/pages/online/github-black.svg" alt="github-black">
                                    <span>GitHub</span>
                                </th>
                                <th>
                                    <img src="/assets/images/pages/common-images/glassdoor.svg" alt="glassdoor">
                                    <span>Glassdoor</span>
                                </th>
                                <th>
                                    <img src="/assets/images/pages/common-images/SimilarWeb.svg" alt="SimilarWeb">
                                    <span>SimilarWeb</span>
                                </th>
                                <th>
                                    <img src="/assets/images/pages/common-images/otta.svg" alt="otta">
                                    <span>Otta</span>
                                </th>
                                <th>
                                    <img src="/assets/images/pages/common/link.svg" alt="link">
                                    <span>Jobs</span>
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            `)
	for _, company := range companies {
		qw422016.N().S(`
                            <tr class="table__item">
                                <td>
                                    <div class="table__item name">
                                        <a class="table__item-link" href="`)
		qw422016.E().S(company.Website)
		qw422016.N().S(`" target="_blank">`)
		qw422016.E().S(company.Name)
		qw422016.N().S(`</a>
                                    </div>
                                </td>
                                <td>
                                    <div class="table__item">
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/square.svg">
                                            <a class="table__item-link" href="https://www.linkedin.com/company/`)
		qw422016.E().S(company.LinkedInProfile.Alias)
		qw422016.N().S(`/" target="_blank" title="`)
		qw422016.E().S(company.LinkedInProfile.Name)
		qw422016.N().S(`">Overview</a>
                                        </div>
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/response.svg">
                                            <a class="table__item-link" href="`)
		qw422016.E().S(linkedinConnectionsURL([]Company{company}, universities))
		qw422016.N().S(`" target="_blank" title="`)
		qw422016.E().S(company.LinkedInProfile.Name)
		qw422016.N().S(`">Connections</a>
                                        </div>
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/vacancy/briefcase.svg">
                                            <a class="table__item-link" href="`)
		qw422016.E().S(linkedinJobsURL([]Company{company}, golangKeywordsTitles))
		qw422016.N().S(`" target="_blank" title="`)
		qw422016.E().S(company.LinkedInProfile.Name)
		qw422016.N().S(`">Jobs</a>
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <div class="table__item">
`)
		if company.GitHubProfile.Login != "" {
			qw422016.N().S(`                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/square.svg">
                                            <a class="table__item-link" href="https://github.com/`)
			qw422016.E().S(company.GitHubProfile.Login)
			qw422016.N().S(`" target="_blank">Overview</a>
                                        </div>
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/database.svg">
                                            <a class="table__item-link" href="https://github.com/orgs/`)
			qw422016.E().S(company.GitHubProfile.Login)
			qw422016.N().S(`/repositories?q=lang:go" target="_blank">Repositories</a>&nbsp;(`)
			qw422016.N().D(fetchGitHubRepositoriesCount(company, Go))
			qw422016.N().S(`)
                                        </div>
`)
		}
		qw422016.N().S(`                                    </div>
                                </td>
                                <td>
                                    <div class="table__item">
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/square.svg">
                                            <a class="table__item-link" href="`)
		qw422016.E().S(company.GlassdoorProfile.OverviewURL)
		qw422016.N().S(`" target="_blank">Overview</a>
                                        </div>
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/message.svg">
                                            <a class="table__item-link" href="`)
		qw422016.E().S(company.GlassdoorProfile.ReviewsURL)
		qw422016.N().S(`" target="_blank">Reviews</a>
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <div class="table__item">
                                        <div class="table__link-group">
                                            <img src="/assets/images/pages/common/square.svg">
                                            <a class="table__item-link" href="`)
		qw422016.E().S(similarwebURL(company.Website))
		qw422016.N().S(`">Overview</a>
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <div class="table__item">
`)
		if company.OttaProfileSlug != "" {
			qw422016.N().S(`                                            <div class="table__link-group">
                                                <img src="/assets/images/pages/common/square.svg">
                                                <a class="table__item-link" href="https://app.otta.com/companies/`)
			qw422016.E().S(company.OttaProfileSlug)
			qw422016.N().S(`" target="_blank">Overview</a>
                                            </div>
`)
		}
		qw422016.N().S(`                                    </div>
                                </td>
                                <td>
                                    <div class="table__item">
`)
		for i, vacancy := range company.Languages[Go].Vacancies {
			qw422016.N().S(`                                            <a class="table__item-link vacancies" href="`)
			qw422016.E().S(vacancy.URL)
			qw422016.N().S(`" target="_blank">Vacancy #`)
			qw422016.N().D(i)
			qw422016.N().S(`</a>
`)
		}
		qw422016.N().S(`                                    </div>
                                </td>
                            </tr>
                            `)
	}
	qw422016.N().S(`
                        </tbody>
                    </table>
                </div>
                <div class="organizer__linkedin">
                    <h2 class="headline headline--lvl1 organizer__block-title">LinkedIn</h2>
                    <div class="organizer__links">
                        <a class="organizer__link" href="`)
	qw422016.E().S(linkedinConnectionsURL(companies, nil))
	qw422016.N().S(`" target="_blank">LinkedIn Connections [Companies]</a>
                        `)
	if len(universities) > 0 {
		qw422016.N().S(`
                            <a class="organizer__link" href="`)
		qw422016.E().S(linkedinConnectionsURL(companies, universities))
		qw422016.N().S(`" target="_blank">LinkedIn Connections [Companies] [Universities]</a>
                        `)
	}
	qw422016.N().S(`
                        <a class="organizer__link" href="`)
	qw422016.E().S(linkedinJobsURL(companies, golangKeywordsTitles))
	qw422016.N().S(`" target="_blank">LinkedIn Jobs [Companies] [Worldwide]</a>
                        <a class="organizer__link" href="`)
	qw422016.E().S(linkedinJobsURL(nil, golangKeywordsTitles))
	qw422016.N().S(`" target="_blank">LinkedIn Jobs [Worldwide]</a>
                    </div>
                </div>
            </div>
        </section>
    </div>
</main>
`)
	streamfooter(qw422016)
	qw422016.N().S(`

</body>
</html>
`)
}

func WriteOrganizerStatic(qq422016 qtio422016.Writer, companies []Company, universities []University) {
	qw422016 := qt422016.AcquireWriter(qq422016)
	StreamOrganizerStatic(qw422016, companies, universities)
	qt422016.ReleaseWriter(qw422016)
}

func OrganizerStatic(companies []Company, universities []University) string {
	qb422016 := qt422016.AcquireByteBuffer()
	WriteOrganizerStatic(qb422016, companies, universities)
	qs422016 := string(qb422016.B)
	qt422016.ReleaseByteBuffer(qb422016)
	return qs422016
}
