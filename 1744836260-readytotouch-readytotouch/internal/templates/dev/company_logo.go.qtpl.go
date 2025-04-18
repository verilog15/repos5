// Code generated by qtc from "company_logo.go.qtpl". DO NOT EDIT.
// See https://github.com/valyala/quicktemplate for details.

package dev

import (
	qtio422016 "io"

	qt422016 "github.com/valyala/quicktemplate"
)

var (
	_ = qtio422016.Copy
	_ = qt422016.AcquireByteBuffer
)

func StreamCompanyLogo(qw422016 *qt422016.Writer, pairs []*CompanyLogoPair, version int) {
	qw422016.N().S(`// Code generated by make generate-organizers. DO NOT EDIT.

package organizers

var (
	CompanyAliasToLogoMapV`)
	qw422016.N().D(version)
	qw422016.N().S(` = map[string]string{
`)
	for _, pair := range pairs {
		qw422016.N().S(` `)
		qw422016.N().S(`"`)
		qw422016.E().S(pair.Alias)
		qw422016.N().S(`":"`)
		qw422016.E().S(pair.Logo)
		qw422016.N().S(`",`)
		qw422016.N().S(`
`)
	}
	qw422016.N().S(`
	}
)
`)
}

func WriteCompanyLogo(qq422016 qtio422016.Writer, pairs []*CompanyLogoPair, version int) {
	qw422016 := qt422016.AcquireWriter(qq422016)
	StreamCompanyLogo(qw422016, pairs, version)
	qt422016.ReleaseWriter(qw422016)
}

func CompanyLogo(pairs []*CompanyLogoPair, version int) string {
	qb422016 := qt422016.AcquireByteBuffer()
	WriteCompanyLogo(qb422016, pairs, version)
	qs422016 := string(qb422016.B)
	qt422016.ReleaseByteBuffer(qb422016)
	return qs422016
}
